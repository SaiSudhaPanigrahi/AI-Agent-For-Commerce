
import os, json, requests
from typing import Dict, Tuple, Any

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

PLANNER_SYSTEM = (
    "You are a shopping agent planner. Choose a tool and strictly return JSON with keys: tool, args. "
    "Tools: 'recommend' (args {query:string, top_k?:int}), 'chitchat' (args {message:string}). "
    "Examples: {\"tool\":\"recommend\",\"args\":{\"query\":\"red running shoes under $80\",\"top_k\":8}} "
    "or {\"tool\":\"chitchat\",\"args\":{\"message\":\"What can you do?\"}}. "
    "Never include extra text."
)

ANSWER_SYSTEM = (
    "You are Mercury, a concise shopping assistant. Use the provided PRODUCT CONTEXT to ground your answer. "
    "Do NOT invent products not present in the context."

)

def _ollama_chat(messages: list[dict]) -> str:
    url = f"{OLLAMA_BASE}/api/chat"
    resp = requests.post(url, json={"model": OLLAMA_MODEL, "messages": messages, "stream": False}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        # new-ish Ollama responses
        msg = data.get("message", {}).get("content") or data.get("response") or ""
        return msg
    return ""

def plan_with_llm(user_msg: str) -> Tuple[str, Dict[str, Any], str]:
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = _ollama_chat(messages)
    try:
        data = json.loads(raw.strip())
    except Exception:
        # fallback: naive JSON detection
        import re
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {"tool":"recommend","args":{"query":user_msg,"top_k":8}}
    tool = data.get("tool", "recommend")
    args = data.get("args", {})
    if tool == "recommend" and "query" not in args:
        args["query"] = user_msg
    if tool == "recommend" and "top_k" not in args:
        args["top_k"] = 8
    if tool == "chitchat" and "message" not in args:
        args["message"] = user_msg
    return tool, args, "ollama"

def respond_with_llm(user_msg: str, context_text: str) -> str:
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"USER: {user_msg}\n\nPRODUCT CONTEXT:\n{context_text}"},
    ]
    raw = _ollama_chat(messages)
    return raw.strip()
