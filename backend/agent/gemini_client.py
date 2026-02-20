from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gemini-2.5-flash"


def _normalize_model_name(name: str) -> str:
    raw = (name or "").strip()
    if raw.startswith("models/"):
        return raw.split("/", 1)[1]
    return raw


@lru_cache(maxsize=32)
def _resolve_model_name(api_key: str, preferred: str) -> str:
    genai.configure(api_key=api_key)
    preferred = _normalize_model_name(preferred) or DEFAULT_MODEL
    fallback_candidates = [
        preferred,
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
    ]

    try:
        available = []
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" not in methods:
                continue
            name = _normalize_model_name(getattr(model, "name", ""))
            if name:
                available.append(name)

        if not available:
            return preferred

        for candidate in fallback_candidates:
            if candidate in available:
                return candidate
        return available[0]
    except Exception:
        return preferred


def get_gemini(model_name: str = DEFAULT_MODEL):
    load_dotenv(dotenv_path=APP_DIR / ".env", override=False)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing. Set it in .env or environment.")
    genai.configure(api_key=api_key)
    resolved_model = _resolve_model_name(api_key, model_name)

    # Gemini will fill these fields (no regex in our code).
    tools = [{
        "function_declarations": [
            {
                "name": "search_text",
                "description": "Search catalog with text query and strict filters",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "q": {"type":"STRING","description":"Include purpose keywords: running, winter, travel, etc."},
                        "category": {"type":"STRING","description":"bags|shoes|jackets|caps"},
                        "color": {"type":"STRING","description":"red|green|blue|black|white|..."},
                        "min_price": {"type":"NUMBER"},
                        "max_price": {"type":"NUMBER"},
                        "k": {"type":"INTEGER"}
                    }
                },
            }
        ]
    }]

    system_instruction = (
        "You are a shopping assistant. When the user asks for products or mentions a category, "
        "ALWAYS call the search_text function with structured arguments. Parse phrases like "
        "'under 100', 'between 50 and 120', 'for running', 'for winter'. Include purpose words "
        "in q. Keep natural replies concise."
    )

    return genai.GenerativeModel(
        model_name=resolved_model,
        tools=tools,
        system_instruction=system_instruction,
        generation_config={"temperature": 0.7}
    )


def get_gemini_smalltalk(model_name: str | None = None):
    load_dotenv(dotenv_path=APP_DIR / ".env", override=False)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing. Set it in .env or environment.")
    genai.configure(api_key=api_key)

    preferred = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    resolved_model = _resolve_model_name(api_key, preferred)
    return genai.GenerativeModel(
        model_name=resolved_model,
        generation_config={"temperature": 0.7, "max_output_tokens": 256},
        system_instruction=(
            "You are Mercury, a friendly, concise shopping assistant for a small catalog. "
            "For chit-chat like 'who are you', 'what can you do', 'what items do you have', or greetings, "
            "answer in 1–2 short paragraphs or a short list. Be creative but do not invent products; "
            "describe real capabilities and categories."
        ),
    )
