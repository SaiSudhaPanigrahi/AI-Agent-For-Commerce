import os, json, re
from urllib.parse import urlparse, parse_qs, unquote
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from schemas import ChatRequest, ChatResponse, RecommendRequest, RecommendResponse, ImageSearchResponse, CatalogResponse, Product
from services.chat import openai_chat, local_smalltalk, has_openai
from services.recommender import TextRecommender
from services.image_search import ImageSearch

app = FastAPI(title="AI Commerce Agent API", version="1.0.4")

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

_recommender = TextRecommender(CATALOG)
_image_search = ImageSearch(CATALOG)

CATEGORY_SYNS = {
    "tshirt":"t-shirts","tshirts":"t-shirts","t-shirt":"t-shirts","t-shirts":"t-shirts","tee":"t-shirts","tees":"t-shirts","shirt":"t-shirts","shirts":"t-shirts",
    "sneaker":"shoes","sneakers":"shoes","shoe":"shoes","shoes":"shoes",
    "short":"shorts","shorts":"shorts",
    "hoodie":"hoodies","hoodies":"hoodies",
    "tank":"tanks","tanks":"tanks",
    "legging":"leggings","leggings":"leggings",
    "bottle":"bottles","bottles":"bottles",
    "bag":"bags","bags":"bags","backpack":"bags","backpacks":"bags"
}

def parse_price_limit(msg: str) -> Optional[float]:
    m = msg.lower()
    m1 = re.search(r"(under|below|less than|<=)\s*\$?\s*(\d+(?:\.\d+)?)", m)
    if m1: return float(m1.group(2))
    m2 = re.search(r"\$\s*(\d+(?:\.\d+)?)", m)
    if m2: return float(m2.group(1))
    m3 = re.search(r"\b(\d{2,4})\s*(bucks|dollars|usd)\b", m)
    if m3: return float(m3.group(1))
    return None

def parse_category(msg: str) -> Optional[str]:
    tokens = re.findall(r"[a-zA-Z\-]+", msg.lower())
    for t in tokens:
        if t in CATEGORY_SYNS: return CATEGORY_SYNS[t]
    return None

def is_shopping_query(msg: str) -> bool:
    m = msg.lower()
    base = ["tshirt","t-shirt","tee","shirt","sneaker","shoe","short","hoodie","tank","legging","bottle","backpack","bag","sports","running","gym","workout","trail","under","below","cheap","budget","dollars","bucks"]
    return any(k in m for k in base)

def shortlist(items: List[Dict[str, Any]], n: int = 6) -> List[Dict[str, Any]]:
    return items[:n]


def bullet_list(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "No matching items found in the catalog."
    out = ["Here are some picks:"]
    for p in items:
        price = f"${p.get('price', 0):.2f}"
        out.append(f"- {p.get('title')} — {price} ({(p.get('brand') or '')} · {(p.get('category') or '')})")
    return "\n".join(out)


def extract_image_url(raw: str) -> Optional[str]:
    m = re.search(r'(https?://\S+)', raw)
    if not m: return None
    url = m.group(1)
    try:
        u = urlparse(url)
        if u.netloc.endswith("google.com"):
            qs = parse_qs(u.query)
            if "imgurl" in qs and qs["imgurl"]:
                return unquote(qs["imgurl"][0])
        return url
    except Exception:
        return url

def route_recommendation(msg: str) -> List[Dict[str, Any]]:
    price_cap = parse_price_limit(msg)
    cat = parse_category(msg)
    results = _recommender.search(msg, top_k=16)
    if cat:
        results = [p for p in results if (p.get("category") or "").lower() == cat] or [p for p in CATALOG if (p.get("category") or "").lower() == cat]
    if price_cap is not None:
        results = [p for p in results if float(p.get("price", 10**9)) <= price_cap] or [p for p in CATALOG if float(p.get("price", 10**9)) <= price_cap]
    if not results:
        results = _recommender.search(" ".join([cat or "", msg]), top_k=16)
    return shortlist(results, n=6)

@app.get("/health")
def health():
    return {"status":"ok","catalog_items": len(CATALOG)}

@app.get("/api/catalog", response_model=CatalogResponse)
def get_catalog() -> CatalogResponse:
    return CatalogResponse(items=[Product(**p) for p in CATALOG])

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    msg = req.message.strip()
    try:
        if has_openai():
            reply = openai_chat(msg)
            return ChatResponse(reply=reply, mode="openai")
        img_url = extract_image_url(msg)
        if img_url:
            try:
                items = _image_search.search_by_image_url(image_url=img_url, top_k=8)
                return ChatResponse(reply=bullet_list(items), mode="local-lite", items=[Product(**p) for p in items])
            except Exception:
                return ChatResponse(reply="Could not process that image URL. Try a direct image link ending in .jpg or .png.", mode="local-lite")
        if is_shopping_query(msg) or parse_price_limit(msg) or parse_category(msg):
            items = route_recommendation(msg)
            return ChatResponse(reply=bullet_list(items), mode="local-lite", items=[Product(**p) for p in items])
        reply = local_smalltalk(msg)
        return ChatResponse(reply=reply, mode="local-lite")
    except Exception as e:
        return ChatResponse(reply=f"Error: {e}", mode="local-lite")

@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    items = _recommender.search(req.query, top_k=req.top_k)
    return RecommendResponse(items=[Product(**p) for p in items])

@app.post("/api/image-search", response_model=ImageSearchResponse)
def image_search(image_url: str = Form(...)) -> ImageSearchResponse:
    url = extract_image_url(image_url) or image_url
    items = _image_search.search_by_image_url(image_url=url, top_k=8)
    return ImageSearchResponse(items=[Product(**p) for p in items])
