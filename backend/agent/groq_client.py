from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parents[1]
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _resolve_model_name() -> str:
    model = (os.getenv("GROQ_MODEL", "") or "").strip()
    return model or DEFAULT_GROQ_MODEL


def groq_compare_completion(prompt: str) -> str:
    load_dotenv(dotenv_path=APP_DIR / ".env", override=False)
    api_key = (os.getenv("GROQ_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing. Set it in backend/.env or environment.")

    payload = {
        "model": _resolve_model_name(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Mercury, an e-commerce compare assistant. "
                    "Return concise, practical recommendations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 700,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(GROQ_BASE_URL, headers=headers, json=payload, timeout=45)
    response.raise_for_status()
    data = response.json() if response.content else {}

    try:
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except Exception:
        return ""
