from __future__ import annotations
import os
import google.generativeai as genai

from dotenv import load_dotenv  # <-- NEW

def get_gemini(model_name: str = "gemini-1.5-flash"):
    # Ensure .env is loaded even if app forgot
    load_dotenv(override=False)  # <-- NEW

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing. Set it in .env or environment.")
    genai.configure(api_key=api_key)

    tools = [{
        "function_declarations": [
            {
                "name": "search_text",
                "description": "Search catalog with text query and strict filters",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "q": {"type":"STRING"},
                        "category": {"type":"STRING"},
                        "color": {"type":"STRING"},
                        "min_price": {"type":"NUMBER"},
                        "max_price": {"type":"NUMBER"},
                        "k": {"type":"INTEGER"}
                    }
                }
            },
            {
                "name": "search_image",
                "description": "Search catalog by a previously uploaded image (server managed)",
                "parameters": {"type":"OBJECT","properties":{"k":{"type":"INTEGER"}}}
            }
        ]
    }]

    return genai.GenerativeModel(
        model_name=model_name,
        tools=tools,
        generation_config={"temperature": 0.7}
    )
