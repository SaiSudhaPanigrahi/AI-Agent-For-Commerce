import os
import re
from typing import Optional
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are Mercury, a friendly shopping agent. "
    "Capabilities: general chat, text recommendations from catalog, image-based product matching. "
    "Be concise and helpful; if the user expresses a shopping intent, present a short list of product picks."
)

def has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

def openai_client() -> Optional[OpenAI]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def openai_chat(message: str) -> str:
    client = openai_client()
    if client is None:
        raise RuntimeError("No OPENAI_API_KEY")
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"),
        messages=[
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": message}
        ],
        temperature=0.5
    )
    return resp.choices[0].message.content.strip()

def _norm(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def local_smalltalk(message: str) -> str:
    m = _norm(message)

    if m in {"hi","hey","hello","yo","hi there","hello there"} or any(w in m for w in ["good morning","good evening","good afternoon"]):
        return "Hi! Tell me what you’re shopping for (e.g., “breathable running tee under $30”)."

    if any(kw in m for kw in ["what can you do","what can u do","capabilities","how can you help","help me","what do you do","what are you able","what are your abilities"]):
        return "I can chat, recommend products from our catalog based on your text, and find visually similar items from an image URL."

    if any(kw in m for kw in ["who are you","who r you","who you are","introduce yourself","tell me about you"]):
        return "I’m Mercury—your shopping agent for this catalog. Ask me for items, budgets, or paste an image to search visually."

    if any(kw in m for kw in ["what is your name","whats your name","what's your name","your name","name please","who are u"]):
        return "I'm Mercury. I can recommend products from the catalog and find similar items from a photo URL."

    return "Try a shopping request like: “lightweight running tee under $30” or paste an image URL."
