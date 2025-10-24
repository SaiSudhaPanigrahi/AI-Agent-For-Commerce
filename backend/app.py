from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Any, Dict, Optional
import os, tempfile, json

from dotenv import load_dotenv  # <-- NEW

from services.catalog_loader import ensure_catalog
from services.text_index import TextIndex
from services.vision_search import VisionIndex
from agent.agent import Agent

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
CACHE_DIR = APP_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load .env early so child processes see it too
load_dotenv(dotenv_path=APP_DIR / ".env", override=False)  # <-- NEW

app = FastAPI(title="AI Commerce Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Bootstrap
catalog_path = ensure_catalog(DATA_DIR, CACHE_DIR, regenerate=False)
text_index = TextIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, force_rebuild=False)
vision_index = VisionIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, data_dir=DATA_DIR, force_rebuild=False)
agent = Agent(text_index=text_index, vision_index=vision_index)

@app.get("/api/catalog")
def get_catalog():
    return JSONResponse(text_index.catalog)

@app.post("/api/reindex")
def reindex():
    global text_index, vision_index, agent
    cat = ensure_catalog(DATA_DIR, CACHE_DIR, regenerate=True)
    text_index = TextIndex(catalog_path=cat, cache_dir=CACHE_DIR, force_rebuild=True)
    vision_index = VisionIndex(catalog_path=cat, cache_dir=CACHE_DIR, data_dir=DATA_DIR, force_rebuild=True)
    agent = Agent(text_index=text_index, vision_index=vision_index)
    return {"ok": True}

# -------- Flexible text search ----------
def _pick_first(d: Dict[str, Any], keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default

def _to_float(x) -> Optional[float]:
    if x is None: return None
    try: return float(x)
    except: return None

def _to_int(x, default: int) -> int:
    try: return int(x)
    except: return default

@app.post("/api/search_text")
async def search_text(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    q = _pick_first(body, ["q", "query", "message", "text", "prompt"], default="")
    filters = body.get("filters") or {}
    if not isinstance(filters, dict):
        filters = {}

    category = body.get("category") or filters.get("category")
    color = body.get("color") or filters.get("color")

    min_price = _to_float(_pick_first(body, ["min_price", "minPrice", "priceMin"], None))
    max_price = _to_float(_pick_first(body, ["max_price", "maxPrice", "priceMax"], None))
    k = _to_int(_pick_first(body, ["k", "topK", "limit"], 12), 12)

    items = text_index.search_with_filters(
        q, category=category, color=color,
        min_price=min_price, max_price=max_price, top_k=k
    )
    return {"results": items, "filters": {"category": category, "color": color}, "q": q, "k": k}

@app.post("/api/search_image")
async def search_image(file: UploadFile = File(...), k: int = Form(8)):
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[-1] or ".jpg")
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    try:
        items = vision_index.search_image_path(Path(tmp_path), top_k=int(k))
    finally:
        try: os.remove(tmp_path)
        except: pass
    return {"results": items}

@app.post("/api/search_by_url")
async def search_by_url(req: Request):
    body = await req.json()
    url = body.get("url")
    k = int(body.get("k", 8))
    items = vision_index.search_image_url(url, top_k=k)
    return {"results": items}

@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    message = str(body.get("message", "")).strip()
    plan = await agent.chat(message)
    return plan
