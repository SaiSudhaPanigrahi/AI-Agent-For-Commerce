import os, re
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers.SentenceTransformer import SentenceTransformer

def chunk_text(s: str, max_words: int = 120, overlap: int = 20) -> List[str]:
    toks = re.findall(r"\w+|\S", s)
    chunks = []
    i = 0
    n = len(toks)
    while i < n:
        j = min(n, i + max_words)
        chunk = " ".join(toks[i:j])
        if chunk.strip():
            chunks.append(chunk)
        if j >= n: break
        i = j - overlap
        if i < 0: i = 0
    return chunks

class RAG:
    def __init__(self, products: List[Dict[str, Any]], embed_model_name: str = "all-MiniLM-L6-v2"):
        self.products = products
        self.embed = SentenceTransformer(embed_model_name)
        self.chunks, self.meta = self._build_chunks(products)
        self.emb = self.embed.encode(self.chunks, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)
        tokenized = [c.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _build_chunks(self, products: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str,Any]]]:
        chunks = []
        meta = []
        for p in products:
            pid = str(p.get("id") or p.get("sku") or p.get("title"))
            title = p.get("title","")
            brand = p.get("brand","")
            cat = p.get("category","")
            desc = p.get("description","") or ""
            tags = " ".join(p.get("tags") or [])
            base = f"{title} | {brand} | {cat} | {tags}\n{desc}"
            cs = chunk_text(base, 120, 20)
            for c in cs:
                chunks.append(c)
                meta.append({"product_id": pid, "title": title, "brand": brand, "category": cat})
        return chunks, meta

    def retrieve(self, query: str, k: int = 12, hint_ids: Optional[Set[str]] = None) -> List[Tuple[str, Dict[str, Any], float]]:
        qv = self.embed.encode([query], normalize_embeddings=True).astype("float32")
        sims, idxs = self.index.search(qv, max(k*8, 64))
        idxs = idxs[0].tolist()
        sims = sims[0].tolist()
        scores = {}
        for i, s in zip(idxs, sims):
            scores[i] = scores.get(i, 0.0) + 0.6 * float(s)
        bm = self.bm25.get_scores(query.lower().split())
        order = sorted(range(len(bm)), key=lambda i: bm[i], reverse=True)[:max(k*8,64)]
        for rank, i in enumerate(order, 1):
            scores[i] = scores.get(i, 0.0) + 0.4 * (1.0 / rank)
        if hint_ids:
            for i in list(scores.keys()):
                pid = str(self.meta[i]["product_id"])
                if pid in hint_ids:
                    scores[i] += 0.05
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = [(self.chunks[i], self.meta[i], float(sc)) for i, sc in top]
        return out

    def top_products_from_chunks(self, chunks: List[Tuple[str, Dict[str, Any], float]], top_n: int = 3) -> List[str]:
        agg = {}
        for _, m, s in chunks:
            pid = str(m["product_id"])
            agg[pid] = agg.get(pid, 0.0) + s
        top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [pid for pid,_ in top]
