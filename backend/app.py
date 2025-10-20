import os, json
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from schemas import ChatRequest, ChatResponse, RecommendRequest, RecommendResponse, ImageSearchResponse, CatalogResponse, Product
from services.recommender import HybridRecommender
from services.image_search import ImageSearch
from services.rag import RAG
from services.agent import Agent, Tools

app = FastAPI(title="AI Commerce Agent API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
with open(CATALOG_PATH, "r") as f:
    CATALOG: List[Dict[str, Any]] = json.load(f)

_recommender = HybridRecommender(CATALOG)
_image_search = ImageSearch(CATALOG)
_rag = RAG(CATALOG, embed_model_name="all-MiniLM-L6-v2")
_tools = Tools(CATALOG, _recommender, _image_search, _rag)
_agent = Agent(CATALOG, _tools)

@app.get("/health")
def health():
    return {"status":"ok","catalog_items": len(CATALOG)}

@app.get("/api/catalog", response_model=CatalogResponse)
def get_catalog() -> CatalogResponse:
    return CatalogResponse(items=[Product(**p) for p in CATALOG])

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = _agent.chat(req.message.strip())
    return ChatResponse(reply=result.message, mode=result.mode, items=[Product(**p) for p in result.items] if result.items else None)

@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    items = _recommender.search(req.query, top_k=req.top_k)
    return RecommendResponse(items=[Product(**p) for p in items])

@app.post("/api/image-search", response_model=ImageSearchResponse)
def image_search(image_url: str = Form(...)) -> ImageSearchResponse:
    items = _image_search.search_by_image_url(image_url=image_url, top_k=8)
    return ImageSearchResponse(items=[Product(**p) for p in items])
