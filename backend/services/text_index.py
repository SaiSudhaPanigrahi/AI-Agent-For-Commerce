from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json, pickle, re
import numpy as np
from sentence_transformers import SentenceTransformer

COLOR_SET = set(["red","blue","green","black","white","yellow","brown","gray","purple","orange","assorted"])
CAT_SET = set(["bags","shoes","jackets","caps"])

def _norm_color(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().lower()
    return s if s in COLOR_SET else None

def _norm_cat(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().lower()
    return s if s in CAT_SET else None

def _text_blob(it: Dict[str, Any]) -> str:
    return " ".join(str(x) for x in [
        it.get("title",""), it.get("description",""),
        it.get("category",""), it.get("color","")
    ] if x)

class TextIndex:
    """
    Sentence-Transformer vector store + strict filters for category/color/price.
    """
    def __init__(self, catalog_path: Path, cache_dir: Path, force_rebuild: bool=False):
        self.catalog: List[Dict[str,Any]] = json.loads(Path(catalog_path).read_text())
        self.cache_dir = cache_dir
        self.vec_path = cache_dir / "sent_vecs.npy"
        self.model_path = cache_dir / "sent_model.pkl"  # store model name only

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)

        if force_rebuild or not self.vec_path.exists():
            self._build()
        else:
            self.vecs = np.load(self.vec_path)

    def _build(self):
        texts = [_text_blob(it) for it in self.catalog]
        embs = self.model.encode(texts, normalize_embeddings=True)
        self.vecs = np.asarray(embs, dtype=np.float32)
        np.save(self.vec_path, self.vecs)
        self.model_path.write_text(self.model_name)

    def _apply_filters(self, idxs: List[int], category: Optional[str], color: Optional[str],
                       min_price: Optional[float], max_price: Optional[float]) -> List[int]:
        ccat = _norm_cat(category); ccol = _norm_color(color)
        out = []
        for i in idxs:
            it = self.catalog[i]
            if ccat and _norm_cat(it.get("category")) != ccat:
                continue
            if ccol and _norm_color(it.get("color")) != ccol:
                continue
            p = float(it.get("price", 0))
            if min_price is not None and p < float(min_price): continue
            if max_price is not None and p > float(max_price): continue
            out.append(i)
        return out

    def search_with_filters(self, query: str,
                            category: Optional[str]=None, color: Optional[str]=None,
                            min_price: Optional[float]=None, max_price: Optional[float]=None,
                            top_k: int=12) -> List[Dict[str, Any]]:
        q = query or "popular picks"
        qv = self.model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
        sims = (self.vecs @ qv)

        all_idx = list(range(len(self.catalog)))
        # hard filters
        idx = self._apply_filters(all_idx, category, color, min_price, max_price)

        # if empty, relax only color (keep category strict if provided)
        if not idx and category:
            idx = self._apply_filters(all_idx, category, None, min_price, max_price)

        if not idx:
            return []

        cand = np.array(idx, dtype=np.int32)
        sc = sims[cand]
        order = np.argsort(-sc)[:top_k]

        out: List[Dict[str,Any]] = []
        for j in order:
            i = int(cand[int(j)])
            item = self.catalog[i].copy()
            item["score"] = float(sc[int(j)])
            out.append(item)
        return out
