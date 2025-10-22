# backend/services/semantic_parser.py
import os, json, re
from typing import Any, Dict, Optional
import google.generativeai as genai

# Use a free-tier friendly model; the -8b is fast & cheap.
CANDIDATE_MODELS = [
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.5-flash-002",
    "models/gemini-1.5-flash",
]

SYSTEM_BRIEF = """You are a commerce query parser.
Extract a compact JSON with the user's intent and normalized filters.

Return ONLY JSON with keys:
- intent: one of ["recommend","chat","image_search"]
- filters: object with keys {category, min_price, max_price, color}
  * category in ["bags","caps","jackets","shoes"] or null
  * min_price, max_price are numbers or null (inclusive bounds)
  * color is a lowercase simple token (e.g., "red") or null

Examples:
User: red bag under 80
{"intent":"recommend","filters":{"category":"bags","min_price":null,"max_price":80,"color":"red"}}

User: compare caps vs jackets for winter
{"intent":"recommend","filters":{"category":null,"min_price":null,"max_price":null,"color":null}}

User: what's your name
{"intent":"chat","filters":{"category":null,"min_price":null,"max_price":null,"color":null}}
"""

CATEGORY_MAP = {
    "bag": "bags", "bags": "bags",
    "cap": "caps", "caps": "caps",
    "jacket": "jackets", "jackets": "jackets",
    "shoe": "shoes", "shoes": "shoes",
}

def _safe_json(s: str) -> Optional[Dict[str, Any]]:
    try: return json.loads(s)
    except Exception: return None

class SemanticParser:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self._model = None
        if self.api_key:
            genai.configure(api_key=self.api_key)
            for name in CANDIDATE_MODELS:
                try:
                    self._model = genai.GenerativeModel(name)
                    # quick ping
                    _ = self._model.generate_content("ok", request_options={"timeout": 5})
                    break
                except Exception:
                    continue

    def parse(self, user_text: str) -> Dict[str, Any]:
        """
        Always returns a dict: {"intent":..., "filters":{category,min_price,max_price,color}}
        Falls back to a minimal heuristic if the API is unavailable.
        """
        # 1) Try Gemini (preferred)
        if self._model:
            prompt = f"{SYSTEM_BRIEF}\n\nUser:\n{user_text}\n\nJSON:"
            try:
                resp = self._model.generate_content(prompt, request_options={"timeout": 20})
                text = (getattr(resp, "text", None) or "").strip()
                data = _safe_json(text)
                if isinstance(data, dict) and "intent" in data and "filters" in data:
                    # Normalize category spelling if model returns singular
                    cat = (data["filters"] or {}).get("category")
                    if cat in CATEGORY_MAP:
                        data["filters"]["category"] = CATEGORY_MAP[cat]
                    return data
            except Exception:
                pass

        # 2) Fallback (tiny heuristic as a safety net)
        ql = user_text.lower()
        intent = "recommend" if any(w in ql for w in
            ["find","show","recommend","under","over","between","bag","cap","jacket","shoe","shoes","caps","bags","jackets"]) else "chat"

        cat = None
        for k, v in CATEGORY_MAP.items():
            if re.search(rf"\b{k}\b", ql):
                cat = v; break

        def _num(pat):
            m = re.search(pat, ql, re.I)
            return float(m.group(1)) if m else None
        max_price = _num(r"(?:under|below|less than|<=|≤)\s*\$?\s*(\d+(?:\.\d+)?)")
        min_price = _num(r"(?:over|above|more than|>=|≥)\s*\$?\s*(\d+(?:\.\d+)?)")
        m_between = re.search(r"(?:between|from)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*\$?\s*(\d+(?:\.\d+)?)", ql)
        if m_between:
            a, b = float(m_between.group(1)), float(m_between.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            min_price, max_price = lo, hi

        color = None
        for c in ["black","white","red","blue","green","yellow","orange","purple","pink","brown","grey","gray","beige","navy","teal"]:
            if re.search(rf"\b{c}\b", ql): color = "grey" if c=="gray" else c; break

        return {"intent": intent, "filters": {"category": cat, "min_price": min_price, "max_price": max_price, "color": color}}
