from __future__ import annotations
import os, asyncio, re
from typing import Any, Dict, Optional, List

from agent.gemini_client import get_gemini
from services.text_index import TextIndex
from services.vision_search import VisionIndex

AGENT_NAME = os.getenv("AGENT_NAME", "Mercury")

def _format_results_text(items: List[Dict[str, Any]], max_n: int = 5) -> str:
    if not items:
        return "I couldn’t find matching items."
    lines = []
    for it in items[:max_n]:
        name = it.get("title") or "Item"
        price = float(it.get("price", 0.0))
        color = it.get("color")
        cat = it.get("category")
        lines.append(f"• {name} — ${price:.2f} ({color}, {cat})")
    extra = "" if len(items) <= max_n else f"\n…and {len(items)-max_n} more."
    return "\n".join(lines) + extra

def _extract_prices_from_text(t: str) -> tuple[Optional[float], Optional[float]]:
    """Strict, tiny fallback so 'under 75' is enforced even if Gemini omits max_price."""
    if not t:
        return (None, None)
    q = t.lower()

    m = re.search(r"between\s*\$?(\d+)\s*(?:and|to)\s*\$?(\d+)", q)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return (lo, hi)

    m = re.search(r"(under|less than|below)\s*\$?(\d+)", q)
    if m:
        return (None, float(m.group(2)))

    m = re.search(r"(over|more than|above)\s*\$?(\d+)", q)
    if m:
        return (float(m.group(2)), None)

    return (None, None)

class Agent:
    """
    Agentic loop using Gemini tool-calling:
    - Gemini extracts q/category/color/min_price/max_price/k and calls search_text.
    - We still enforce price caps server-side (≤ max) even if Gemini omits them.
    - Reply includes a readable text list for the chat bubble.
    """
    def __init__(self, text_index: TextIndex, vision_index: VisionIndex):
        self.text_index = text_index
        self.vision_index = vision_index
        self.model = None  # lazy init

    def _ensure_model(self):
        if self.model is None:
            self.model = get_gemini()

    def _maybe_name(self, t: str) -> Optional[str]:
        t2 = (t or "").lower().strip()
        if any(p in t2 for p in ["what is your name","what's your name","who are you"]):
            return f"My name is {AGENT_NAME}. I can chat, recommend products (try “red bags under $50”), and do image search—upload a photo!"
        return None

    async def chat(self, user_text: str) -> Dict[str, Any]:
        # quick deterministic path for "what's your name"
        nm = self._maybe_name(user_text)
        if nm:
            return {"intent": "chat", "text": nm, "reply": nm, "results": [], "filters": {}}

        # Let Gemini parse & tool-call
        try:
            self._ensure_model()
            resp = await asyncio.to_thread(
                self.model.generate_content,
                [{"role":"user","parts":[user_text]}]
            )
            if resp and resp.candidates:
                cand = resp.candidates[0]
                parts = cand.content.parts if getattr(cand, "content", None) else []
                for p in parts:
                    fc = getattr(p, "function_call", None)
                    if fc and fc.name == "search_text":
                        args = fc.args or {}
                        q = (args.get("q") or user_text or "").strip()
                        category = args.get("category")
                        color = args.get("color")
                        min_price = args.get("min_price")
                        max_price = args.get("max_price")
                        k = int(args.get("k") or 8)

                        # STRICT price cap fallback from the natural language if Gemini omitted it
                        lo_fallback, hi_fallback = _extract_prices_from_text(user_text)
                        if min_price is None and lo_fallback is not None:
                            min_price = lo_fallback
                        if max_price is None and hi_fallback is not None:
                            max_price = hi_fallback

                        items = self.text_index.search_with_filters(
                            q, category=category, color=color,
                            min_price=min_price, max_price=max_price, top_k=k
                        )

                        # Double-guard: remove any > max_price just in case
                        if max_price is not None:
                            items = [it for it in items if float(it.get("price", 0.0)) <= float(max_price)]

                        reply = "Here are some options:\n" + _format_results_text(items)
                        return {"intent":"recommend","reply":reply,"text":reply,"results":items,
                                "filters":{"category":category,"color":color}}

            # If Gemini didn’t tool-call, just return its text
            if resp and resp.text:
                msg = resp.text.strip()
                return {"intent":"chat","text":msg,"reply":msg,"results":[],"filters":{}}

        except Exception:
            # Soft-fail into deterministic RAG if Gemini unavailable
            pass

        # Deterministic last resort with strict price extraction too
        lo_fallback, hi_fallback = _extract_prices_from_text(user_text)
        items = self.text_index.search_with_filters(
            user_text, min_price=lo_fallback, max_price=hi_fallback, top_k=8
        )
        if hi_fallback is not None:
            items = [it for it in items if float(it.get("price", 0.0)) <= float(hi_fallback)]
        if items:
            reply = "Here are some options:\n" + _format_results_text(items)
            return {"intent":"recommend","text":reply,"reply":reply,"results":items,"filters":{}}

        msg = f"Hi! I’m {AGENT_NAME}. Ask for things like “shoes under $75” or “caps for winter under $30”."
        return {"intent":"chat","text":msg,"reply":msg,"results":[],"filters":{}}
