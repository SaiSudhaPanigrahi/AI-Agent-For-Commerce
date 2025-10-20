import os, json, re, requests
from typing import Dict, Any, Optional, Tuple, List

def has_ollama() -> bool:
    return bool(os.environ.get("OLLAMA_BASE_URL")) and bool(os.environ.get("OLLAMA_MODEL"))

def _ollama_chat(messages: List[Dict[str,str]], temperature: float = 0.2, json_mode: bool = False) -> str:
    url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b-instruct")
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
    if json_mode:
        payload["format"] = "json"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")

def plan_with_llm(user_msg: str) -> Tuple[str, Dict[str, Any]]:
    sys = "You decide which single tool to use: text-recommend, image-search, smalltalk, qa-rag. Use image-search only if a direct image URL is present. For qa-rag, the user is asking to compare, explain, justify, or ask product questions. Respond as strict JSON: {\"tool\":\"...\",\"args\":{...}}. Valid args: for text-recommend: {\"query\": \"...\"}; for image-search: {\"image_url\":\"...\"}; for smalltalk: {\"message\":\"...\"}; for qa-rag: {\"question\":\"...\"}."
    usr = f"User: {user_msg}"
    out = _ollama_chat([{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.1, json_mode=False)
    j = None
    try:
        j = json.loads(out)
    except Exception:
        m = re.search(r"\{.*\}", out, re.S)
        if m:
            try:
                j = json.loads(m.group(0))
            except Exception:
                j = None
    if isinstance(j, dict) and "tool" in j and "args" in j:
        tool = j["tool"]
        args = j["args"]
        if tool == "image-search" and "image_url" not in args:
            m2 = re.search(r'(https?://[^\s]+?\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s]*)?)', user_msg, re.I)
            if m2:
                args["image_url"] = m2.group(1)
        return tool, args
    return "smalltalk", {"message": user_msg}

def answer_with_llm(question: str, context: str) -> str:
    sys = "You answer using only the provided context. Be concise. Use bullet points. Do not mention any item not present in the context. If unsure, say what info is missing."
    usr = f"Question:\n{question}\n\nContext:\n{context}"
    return _ollama_chat([{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.2, json_mode=False)
