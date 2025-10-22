
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
try:
    import faiss  # faiss-cpu (optional)
except Exception:
    faiss = None

from .utils import cosine_sim

_WORD = re.compile(r"\w+|\S")

def _product_text(p: Dict[str, Any]) -> str:
    parts = [p.get("title",""), p.get("brand",""), p.get("category",""), p.get("color",""), p.get("description","") ]
    return " ".join([str(x) for x in parts if x])

def _tokenize(s: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(s)]

class RAGIndex:
    def __init__(self, products: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2"):
        self.products = products
        self.chunks: List[str] = []
        self.meta: List[Tuple[int,int]] = []  # (product_idx, chunk_id)
        for i, p in enumerate(products):
            text = _product_text(p)
            self.chunks.append(text)
            self.meta.append((i, 0))
        # BM25
        self.bm25 = BM25Okapi([_tokenize(c) for c in self.chunks])
        # Embeddings
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode(self.chunks, convert_to_numpy=True, normalize_embeddings=True)
        # FAISS index optional
        if faiss is not None:
            d = self.emb.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.emb.astype('float32'))
        else:
            self.index = None

    def _semantic_top(self, q: str, k: int) -> List[int]:
        qv = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        if self.index is not None:
            D, I = self.index.search(qv.astype('float32'), k)
            return I[0].tolist()
        sims = cosine_sim(qv, self.emb)[0]
        order = np.argsort(-sims)[:k]
        return order.tolist()

    def search_products(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = np.argsort(-bm25_scores)[:k*2]
        sem_top = self._semantic_top(query, k*2)
        # fuse
        seen = set()
        ordered = []
        for idx in list(bm25_top) + list(sem_top):
            if idx not in seen and 0 <= idx < len(self.meta):
                seen.add(idx)
                prod_idx, _ = self.meta[idx]
                if prod_idx not in [o.get('id') for o in ordered]:
                    ordered.append(self.products[prod_idx])
        return ordered[:k]
