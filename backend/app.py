# backend/app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import os

from services.catalog_loader import ensure_catalog
from services.embedding_index import EmbeddingIndex
from services.vision_search import VisionIndex
from agent.agent import Agent

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Commerce Agent (Gemini + RAG)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve product images
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# ---------- Schemas ----------
class ChatIn(BaseModel):
    message: str

class TextSearchIn(BaseModel):
    query: str
    k: int = 8

class UrlSearchIn(BaseModel):
    url: str
    k: int = 8

# ---------- Bootstrap ----------
catalog_path = ensure_catalog(DATA_DIR, CACHE_DIR)
embed_index = EmbeddingIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR)
vision_index = VisionIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, data_dir=DATA_DIR)
agent = Agent(embed_index=embed_index, vision_index=vision_index)

# ---------- Routes ----------
@app.get("/api/catalog")
def get_catalog():
    return JSONResponse(embed_index.catalog)

@app.post("/api/reindex")
def reindex():
    global embed_index, vision_index, agent
    catalog = ensure_catalog(DATA_DIR, CACHE_DIR, regenerate=True)
    embed_index = EmbeddingIndex(catalog_path=catalog, cache_dir=CACHE_DIR, force_rebuild=True)
    vision_index = VisionIndex(catalog_path=catalog, cache_dir=CACHE_DIR, data_dir=DATA_DIR, force_rebuild=True)
    agent = Agent(embed_index=embed_index, vision_index=vision_index)
    return {"ok": True}

@app.post("/api/chat")
def chat(inp: ChatIn):
    plan = agent.chat(inp.message.strip())  # dict with intent/filters/results/reply
    text = plan.get("reply", "")
    # Keep legacy keys for frontend while returning full plan
    plan["reply"] = text
    plan["text"] = text
    return plan

@app.post("/api/search_text")
def search_text(inp: TextSearchIn):
    items = embed_index.search(inp.query, top_k=inp.k)
    return {"results": items}

# compatibility route for UIs that POST to /api/recommend
@app.post("/api/recommend")
def recommend(inp: TextSearchIn):
    items = embed_index.search(inp.query, top_k=inp.k)
    return {"results": items}

@app.post("/api/search_image")
async def search_image(file: UploadFile = File(...), k: int = Form(8)):
    tmp = CACHE_DIR / ("_upload_" + file.filename.replace("/", "_"))
    with tmp.open("wb") as f:
        f.write(await file.read())
    try:
        items = vision_index.search_image_path(tmp, top_k=k)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass
    return {"results": items}

@app.post("/api/search_by_url")
def search_by_url(inp: UrlSearchIn):
    items = vision_index.search_image_url(inp.url, top_k=inp.k)
    return {"results": items}
