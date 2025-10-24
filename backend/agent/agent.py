from __future__ import annotations
import os, re, asyncio
from typing import Any, Dict, Optional

from agent.gemini_client import get_gemini
from services.text_index import TextIndex
from services.vision_search import VisionIndex

AGENT_NAME = os.getenv("AGENT_NAME", "Mercury")

class Agent:
    def __init__(self, text_index: TextIndex, vision_index: VisionIndex):
        self.text_index = text_index
        self.vision_index = vision_index
        self.model = None  # <-- lazy

    def _ensure_model(self):
        if self.model is None:
            self.model = get_gemini()

    def _parse_price(self, t: str) -> tuple[Optional[float], Optional[float]]:
        lo = hi = None
        m = re.search(r"under\s*\$?(\d+)", t);  hi = float(m.group(1)) if m else None
        m = re.search(r"over\s*\$?(\d+)", t);   lo = float(m.group(1)) if m else None
        return lo, hi

    def _maybe_name(self, t: str) -> Optional[str]:
        t2 = t.lower().strip()
        if any(p in t2 for p in ["what is your name","what's your name","who are you"]):
            return f"My name is {AGENT_NAME}. I can chat, recommend products (try “red bags under $50”), and do image search—upload a photo!"
        return None

    async def chat(self, user_text: str) -> Dict[str, Any]:
        nm = self._maybe_name(user_text)
        if nm:
            return {"intent": "chat", "text": nm, "reply": nm, "results": [], "filters": {}}

        # Try Gemini tool-use; if key is missing, we fall back gracefully
        try:
            self._ensure_model()
            prompt = (
                f"You are {AGENT_NAME}, a helpful shopping assistant. "
                "Decide whether to call search_text or just chat. "
                "If the user mentions a category or color (e.g., red bag), use search_text with strict filters. "
                "Otherwise, answer helpfully. Keep replies concise."
            )
            resp = await asyncio.to_thread(
                self.model.generate_content,
                [{"role":"user","parts":[prompt]},
                 {"role":"user","parts":[user_text]}]
            )
            if resp and resp.candidates:
                cand = resp.candidates[0]
                parts = cand.content.parts if getattr(cand, "content", None) else []
                for p in parts:
                    fc = getattr(p, "function_call", None)
                    if fc and fc.name == "search_text":
                        args = fc.args or {}
                        q = args.get("q") or user_text
                        category = args.get("category")
                        color = args.get("color")
                        min_price = args.get("min_price")
                        max_price = args.get("max_price")
                        k = int(args.get("k") or 8)

                        if min_price is None or max_price is None:
                            lo, hi = self._parse_price(user_text)
                            if min_price is None: min_price = lo
                            if max_price is None: max_price = hi

                        items = self.text_index.search_with_filters(
                            q, category=category, color=color,
                            min_price=min_price, max_price=max_price, top_k=k
                        )
                        reply = "Here are some options I found."
                        return {"intent":"recommend","reply":reply,"text":reply,"results":items,
                                "filters":{"category":category,"color":color}}

            if resp and resp.text:
                msg = resp.text.strip()
                return {"intent":"chat","text":msg,"reply":msg,"results":[],"filters":{}}

        except Exception:
            # fall through to deterministic router
            pass

        # Deterministic router as fallback
        lo, hi = self._parse_price(user_text)
        t = user_text.lower()
        category = None
        for w in ("bags","bag","shoes","shoe","jackets","jacket","caps","cap"):
            if re.search(rf"\b{w}\b", t):
                category = {"bag":"bags","bags":"bags","shoe":"shoes","shoes":"shoes","jacket":"jackets","jackets":"jackets","cap":"caps","caps":"caps"}[w]
                break
        color = None
        for c in ("red","blue","green","black","white","yellow","brown","gray","purple","orange"):
            if re.search(rf"\b{c}\b", t):
                color = c; break

        if category or any(x in t for x in ["recommend","suggest","looking","find","show me","under","over"]):
            items = self.text_index.search_with_filters(user_text, category=category, color=color, min_price=lo, max_price=hi, top_k=8)
            reply = "Here are some options."
            return {"intent":"recommend","text":reply,"reply":reply,"results":items,"filters":{"category":category,"color":color}}

        msg = f"Hi! I’m {AGENT_NAME}. Ask for things like “red bags under $50” or upload a photo to find look-alikes."
        return {"intent":"chat","text":msg,"reply":msg,"results":[],"filters":{}}
