from __future__ import annotations
import os, asyncio, re
from typing import Any, Dict, Optional, List, Tuple

from agent.gemini_client import get_gemini
from services.text_index import TextIndex
from services.vision_search import VisionIndex

import re
from agent.gemini_client import get_gemini, get_gemini_smalltalk  # add smalltalk model

# small-talk recognizers
_SMALLTALK_WHO   = re.compile(r"\b(who are you|what('?| i)s your name|your name\??)\b", re.I)
_SMALLTALK_DO    = re.compile(r"\b(what can you do|how (can|do) you help|what do you do)\b", re.I)
_SMALLTALK_ITEMS = re.compile(r"\b(what (items|products) (do you )?have|what('?| i)s in (the )?catalog|what categories)\b", re.I)
_SMALLTALK_HELLO = re.compile(r"\b(hi|hello|hey|hiya|yo|sup)\b", re.I)

# quick image-url detector (leave file-upload to your /api/search_image endpoint)
_IMG_URL = re.compile(r"https?://\S+\.(png|jpg|jpeg|webp|gif|avif)\b", re.I)


AGENT_NAME = os.getenv("AGENT_NAME", "Mercury")

# ---------------- formatting helpers ----------------

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

# ---------------- tiny NL price/category fallback ----------------

_PRICE_BETWEEN = re.compile(r"between\s*\$?(\d+(?:\.\d+)?)\s*(?:and|to)\s*\$?(\d+(?:\.\d+)?)", re.I)
_PRICE_UNDER   = re.compile(r"(under|less than|below)\s*\$?(\d+(?:\.\d+)?)", re.I)
_PRICE_OVER    = re.compile(r"(over|more than|above)\s*\$?(\d+(?:\.\d+)?)", re.I)

def _extract_prices_from_text(t: str) -> Tuple[Optional[float], Optional[float]]:
    if not t:
        return (None, None)
    m = _PRICE_BETWEEN.search(t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (min(a,b), max(a,b))
    m = _PRICE_UNDER.search(t)
    if m:
        return (None, float(m.group(2)))
    m = _PRICE_OVER.search(t)
    if m:
        return (float(m.group(2)), None)
    return (None, None)

def _extract_category_fallback(t: str) -> Optional[str]:
    s = t.lower()
    if "cap" in s: return "caps"
    if "shoe" in s: return "shoes"
    if "bag" in s: return "bags"
    if "jacket" in s or "coat" in s: return "jackets"
    return None

# ---------------- agent ----------------

class Agent:
    """
    Agentic loop using Gemini tool-calling:
    - Gemini extracts q/category/color/min_price/max_price/k and calls search_text.
    - We enforce price caps server-side (≤ max) even if Gemini omits them.
    - If no results, we clearly say 'Not present' and optionally show the closest alternatives.
    """
    def __init__(self, text_index: TextIndex, vision_index: VisionIndex):
        self.text_index = text_index
        self.vision_index = vision_index
        self.model = None  # lazy init

    def _catalog_overview(self) -> str:
        items = self.text_index.catalog
        cats: Dict[str, int] = {}
        colors: Dict[str, int] = {}
        prices: List[float] = []
        for it in items:
            c = (it.get("category") or "").strip().lower()
            if c: cats[c] = cats.get(c, 0) + 1
            col = (it.get("color") or "").strip().lower()
            if col: colors[col] = colors.get(col, 0) + 1
            try:
                prices.append(float(it.get("price", 0.0)))
            except:
                pass
        cat_line = ", ".join(f"{k} ({v})" for k, v in sorted(cats.items()))
        top_colors = ", ".join(k for k, _ in sorted(colors.items(), key=lambda kv: -kv[1])[:6]) or "various colors"
        price_line = f"${min(prices):.0f}–${max(prices):.0f}" if prices else "n/a"
        return f"Categories: {cat_line or 'none'}. Popular colors: {top_colors}. Price range: {price_line}. Total items: {len(items)}."

    async def _maybe_smalltalk(self, user_text: str) -> Optional[Dict[str, Any]]:
        t = (user_text or "").strip()
        if not (_SMALLTALK_WHO.search(t) or _SMALLTALK_DO.search(t) or _SMALLTALK_ITEMS.search(
                t) or _SMALLTALK_HELLO.search(t)):
            return None

        overview = self._catalog_overview()
        prompt = (
            f"You are {AGENT_NAME}. Here is the current catalog overview:\n{overview}\n\n"
            f"User: {t}\n"
            "Answer in 2–4 sentences, concrete, friendly, and playful."
        )
        try:
            model = get_gemini_smalltalk()
            resp = await asyncio.to_thread(model.generate_content, [prompt])
            txt = (getattr(resp, "text",
                           None) or "").strip() or "I’m an LLM-powered shopping assistant for this catalog."
        except Exception:
            # fallback if API/quota is down
            if _SMALLTALK_WHO.search(t):
                txt = f"I’m {AGENT_NAME}, your AI shopping sidekick."
            elif _SMALLTALK_DO.search(t):
                txt = "I can chat about your needs, extract filters like color and budget, and find matching products. I also support image-based search."
            else:
                txt = "We’ve got a compact catalog of bags, shoes, jackets and caps in multiple colors and prices. Ask me for something specific!"

        txt = f"[LLM] {txt}"
        return {"intent": "smalltalk", "source": "llm", "text": txt, "reply": txt, "results": [], "filters": {}}

    def _ensure_model(self):
        if self.model is None:
            self.model = get_gemini()

    # def _maybe_name(self, t: str) -> Optional[str]:
    #     t2 = (t or "").lower().strip()
    #     if any(p in t2 for p in ["what is your name","what's your name","who are you"]):
    #         return f"My name is {AGENT_NAME}. I can chat, recommend products (try “red bags under $50”), and do image search—upload a photo!"
    #     return None

    async def chat(self, user_text: str) -> Dict[str, Any]:
        # quick deterministic path for "what's your name"
        # nm = self._maybe_name(user_text)
        # if nm:
        #     return {"intent": "chat", "text": nm, "reply": nm, "results": [], "filters": {}}

        # Defaults (we’ll update from Gemini/tool call)
        q = user_text.strip()
        category = _extract_category_fallback(user_text)  # ensure “caps” is respected even if Gemini misses it
        color = None
        min_price, max_price = _extract_prices_from_text(user_text)
        k = 12

        # 0) Image URL in text? Route to vision search (rule-based)
        m = _IMG_URL.search(user_text or "")
        if m:
            try:
                items = self.vision_index.search_image_url(m.group(0), top_k=12)
            except Exception:
                items = []
            if items:
                msg = "[Rule-based] Here are visually similar items:\n" + _format_results_text(items)
            else:
                msg = "[Rule-based] I couldn’t find visually similar items."
            return {"intent": "image_search", "source": "rule-based", "text": msg, "reply": msg, "results": items,
                    "filters": {}}

        # 1) Small-talk? Let the LLM answer creatively.
        small = await self._maybe_smalltalk(user_text)
        if small:
            return small

        # Try Gemini tool-calling first
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
                        q = (args.get("q") or q).strip()
                        # Prefer Gemini’s structured values but keep our strict fallbacks
                        category = args.get("category") or category
                        color = args.get("color") or color
                        if min_price is None and (args.get("min_price") is not None):
                            min_price = args.get("min_price")
                        if max_price is None and (args.get("max_price") is not None):
                            max_price = args.get("max_price")
                        k = int(args.get("k") or k)
                        break
        except Exception:
            pass  # fall back to deterministic path

        # Run search (strict price filters if provided)
        items = self.text_index.search_with_filters(
            q, category=category, color=color,
            min_price=min_price, max_price=max_price, top_k=k
        )

        # Extra guard: remove any > max_price
        if max_price is not None:
            items = [it for it in items if float(it.get("price", 0.0)) <= float(max_price)]

        # If nothing found and a price cap was requested, be vocal and offer closest alternatives
        if not items and (max_price is not None or "under" in user_text.lower()):
            # Pull nearest items in the same category (if specified), sorted by price asc
            pool = [it for it in self.text_index.catalog if (category is None or it.get("category")==category)]
            pool.sort(key=lambda it: float(it.get("price", 0.0)))
            # Take a few just above the budget (within +$15 window)
            near = []
            if max_price is not None:
                for it in pool:
                    p = float(it.get("price", 0.0))
                    if p > float(max_price) and p <= float(max_price) + 15.0:
                        near.append(it)
                        if len(near) >= 5: break

            budget_str = f"${float(max_price):.0f}" if max_price is not None else "your budget"
            cat_str = f" {category}" if category else ""
            reply = f"Not present — I don’t have any{cat_str} under {budget_str}."
            if near:
                reply += "\nClosest options slightly above your budget:\n" + _format_results_text(near)
                return {"intent":"recommend","text":reply,"reply":reply,"results":near,"filters":{"category":category,"color":color}}
            else:
                return {"intent":"chat","text":reply,"reply":reply,"results":[],"filters":{"category":category,"color":color}}

        # Normal path: we have items
        if items:
            # Be explicit if user asked for under-$ and we satisfied it
            prefix = ""
            if max_price is not None:
                prefix = f"Here are {category or 'items'} at or under ${float(max_price):.0f}:\n"
            elif category:
                prefix = f"Here are some {category} matches:\n"
            else:
                prefix = "Here are some options:\n"
            reply = prefix + _format_results_text(items)
            return {"intent":"recommend","text":reply,"reply":reply,"results":items,"filters":{"category":category,"color":color}}

        # No items and no explicit budget — just say so
        msg = "I couldn’t find matching items."
        return {"intent":"chat","text":msg,"reply":msg,"results":[],"filters":{"category":category,"color":color}}
