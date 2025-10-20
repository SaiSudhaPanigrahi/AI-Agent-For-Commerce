from typing import List, Dict, Any, Tuple
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def _build_text(p: Dict[str, Any]) -> str:
    parts = [p.get("title",""), p.get("brand",""), p.get("category",""), p.get("description","")]
    if p.get("tags"):
        parts.append(" ".join(p["tags"]))
    return " | ".join([x for x in parts if x])

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())

class HybridRecommender:
    def __init__(self, products: List[Dict[str, Any]], embed_model_name: str = "all-MiniLM-L6-v2"):
        self.products = products
        self.corpus = [_build_text(p) for p in products]
        self.tokens = [_tokenize(c) for c in self.corpus]
        self.bm25 = BM25Okapi(self.tokens)
        self.model = SentenceTransformer(embed_model_name)
        emb = self.model.encode(self.corpus, normalize_embeddings=True, show_progress_bar=False)
        self.emb = np.asarray(emb, dtype="float32")
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        qv = self.model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")
        N = max(64, top_k * 6)
        sims, idxs = self.index.search(qv, N)
        idxs = idxs[0].tolist()
        sims = sims[0].tolist()
        scores = {}
        for i, s in zip(idxs, sims):
            scores[i] = scores.get(i, 0.0) + 0.6 * float(s)
        bm = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(bm)), key=lambda i: bm[i], reverse=True)[:N]
        for rank, i in enumerate(order, 1):
            scores[i] = scores.get(i, 0.0) + 0.4 * (1.0 / rank)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self.products[i] for i, _ in top]
