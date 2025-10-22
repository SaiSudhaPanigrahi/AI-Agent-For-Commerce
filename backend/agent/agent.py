# backend/agent/agent.py
"""
Agentic router for the commerce app.

- Uses Gemini (free-tier friendly) to parse user intent + filters via SemanticParser.
- Routes to:
    * recommend  -> embedding_index.search_with_filters(...)
    * image_search -> tell frontend to call /api/search_image or /api/search_by_url
    * chat       -> concise general chat (Gemini if available, fallback otherwise)
- Returns a structured plan so the frontend can render one endpoint:
    { intent, filters, results, reply, text }
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from services.semantic_parser import SemanticParser

# Prefer small/fast Gemini variants that work with Google AI Studio keys
CANDIDATE_MODELS = [
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.5-flash-002",
    "models/gemini-1.5-flash",
]


class Agent:
    def __init__(self, embed_index, vision_index=None) -> None:
        """
        :param embed_index: EmbeddingIndex (must implement search_with_filters(query, filters, top_k))
        :param vision_index: optional image search index (not used here; image search has its own endpoints)
        """
        self.embed_index = embed_index
        self.vision_index = vision_index

        # LLM for summarization / general chat (optional; app works offline without it)
        self._model: Optional[genai.GenerativeModel] = None
        try:
            # We don't call genai.configure here; SemanticParser already configured with GOOGLE_API_KEY.
            # If the user set GOOGLE_API_KEY, we can still try to initialize a model for summaries.
            for name in CANDIDATE_MODELS:
                m = genai.GenerativeModel(name)
                # quick ping (short timeout)
                _ = m.generate_content("ok", request_options={"timeout": 5})
                self._model = m
                break
        except Exception:
            self._model = None

        # Semantic parser (Gemini-backed with a safe heuristic fallback)
        self.parser = SemanticParser()

    # ------------------ private helpers ------------------

    def _summarize(self, user_text: str, items: List[Dict[str, Any]]) -> str:
        """
        Create a short, user-friendly recommendation summary.
        Tries Gemini; falls back to a deterministic local summary if unavailable.
        """
        # Gemini summary
        if self._model:
            try:
                prompt = (
                    "You are a concise, friendly shopping assistant.\n"
                    f"User asked: {user_text}\n\n"
                    "Here are candidate items in JSON (title, price, color, category, description):\n"
                    + json.dumps(items[:8], ensure_ascii=False)
                    + "\n\nWrite 3–6 bullets, each like:\n"
                      "• Title — $Price — one short reason\n"
                      "Keep it brief."
                )
                r = self._model.generate_content(prompt, request_options={"timeout": 25})
                txt = getattr(r, "text", None)
                if txt:
                    return txt.strip()
            except Exception:
                pass

        # Offline fallback summary
        if not items:
            return "I didn’t find matching items. Try another color or price range."
        bullets = []
        for it in items[:6]:
            title = it.get("title", "Item")
            price = float(it.get("price", 0.0))
            color = (it.get("color") or "assorted")
            cat = (it.get("category") or "item")
            bullets.append(f"• {title} — ${price:.2f} — solid {color} {cat[:-1]} pick")
        return "Here are a few picks:\n" + "\n".join(bullets)

    # ------------------ public API ------------------

    def chat(self, user_text: str) -> Dict[str, Any]:
        """
        Main entry point. Returns a dict with:
          - intent: "recommend" | "image_search" | "chat"
          - filters: {category, min_price, max_price, color}
          - results: list[Item] (only for recommend; may be empty otherwise)
          - reply: string (human-friendly)
          - text: string (alias of reply for UI compatibility)
        """
        # 1) Ask Gemini (via SemanticParser) to understand the user's intent + normalized filters.
        plan = self.parser.parse(user_text)  # {"intent":..., "filters": {...}}
        intent = plan.get("intent", "chat")
        filters = plan.get("filters", {}) or {}

        # 2) Route by intent
        if intent == "image_search":
            # UI should then call /api/search_image or /api/search_by_url
            reply = "Upload an image via /api/search_image or paste a link via /api/search_by_url to find visually similar items."
            return {"intent": intent, "filters": filters, "results": [], "reply": reply, "text": reply}

        if intent == "recommend":
            # Use explicit filters from Gemini to build the candidate set + rank semantically
            items = self.embed_index.search_with_filters(user_text, filters, top_k=12)
            reply = self._summarize(user_text, items)
            return {"intent": intent, "filters": filters, "results": items, "reply": reply, "text": reply}

        # Default: general chat
        if self._model:
            try:
                r = self._model.generate_content(
                    "You are a helpful, concise shopping assistant.\nUser: " + user_text,
                    request_options={"timeout": 20},
                )
                txt = (getattr(r, "text", "") or "").strip() or "Hi! How can I help you shop today?"
            except Exception:
                txt = "Hi! How can I help you shop today?"
        else:
            txt = "Hi! I can recommend items from our catalog, search by text, or even by an image you upload."

        return {"intent": "chat", "filters": filters, "results": [], "reply": txt, "text": txt}
