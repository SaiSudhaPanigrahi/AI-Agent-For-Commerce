import os
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

def local_smalltalk(message: str) -> str:
    m = message.lower().strip()
    if "name" in m:
        return "I'm Mercury. I can recommend products from the catalog and find similar items from a photo URL."
    if any(k in m for k in ["what can you do","help","capabilities","how can you help"]):
        return "I can: 1) chat, 2) recommend items from our catalog from a text request, 3) find similar items from an image URL."
    if any(k in m for k in ["hi","hello","hey"]):
        return "Hi! Tell me what you’re shopping for (e.g., “breathable running tee under $30”)."
    return "Try a shopping request like: “lightweight running tee under $30” or paste an image URL."
