from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from .utils import cosine_topk

def _build_text(p: Dict[str, Any]) -> str:
    parts = [p.get("title",""), p.get("brand",""), p.get("category",""), p.get("description","")]
    if p.get("tags"):
        parts.append(" ".join(p["tags"]))
    return " | ".join([x for x in parts if x])

class TextRecommender:
    def __init__(self, products: List[Dict[str, Any]]):
        self.products = products
        self.corpus = [_build_text(p) for p in products]
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.emb = self.model.encode(self.corpus, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        idxs, _ = cosine_topk(self.emb, q, top_k)
        return [self.products[i] for i in idxs]
