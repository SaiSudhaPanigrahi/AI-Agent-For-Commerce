from __future__ import annotations
import os
import google.generativeai as genai
from dotenv import load_dotenv

def get_gemini(model_name: str = "gemini-1.5-flash"):
    load_dotenv(override=False)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing. Set it in .env or environment.")
    genai.configure(api_key=api_key)

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
        model_name=model_name,
        tools=tools,
        system_instruction=system_instruction,
        generation_config={"temperature": 0.7}
    )
