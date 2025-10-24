from pathlib import Path
import os, json, time, pickle, re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    # Optional: local TF-IDF fallback if embeddings unavailable
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

# Supported categories and normalization
CATS = {
    "bag": "bags", "bags": "bags",
    "cap": "caps", "caps": "caps",
    "jacket": "jackets", "jackets": "jackets",
    "shoe": "shoes", "shoes": "shoes",
}

COLOR_SET = set(["red","blue","green","black","white","yellow","brown","gray","purple","orange","assorted"])

def _norm_cat(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().lower()
    return CATS.get(s, s if s in CATS.values() else None)

def _norm_color(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().lower()
    return s if s in COLOR_SET else None

def _text_blob(it: Dict[str, Any]) -> str:
    # unified text for embedding/tfidf
    fields = [
        it.get("title",""),
        it.get("description",""),
        it.get("category",""),
        it.get("color",""),
    ]
    return " ".join([str(x) for x in fields if x])

class EmbeddingIndex:
    """
    Text retrieval with strict server-side filters.
    Uses a lightweight TF-IDF index (fully local) for robustness.
    Caches to backend/.cache/.
    """
    def __init__(self, catalog_path: Path, cache_dir: Path, force_rebuild: bool = False):
        self.catalog_path = catalog_path
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.cache_dir / "tfidf_vec.pkl"
        self.mat_path = self.cache_dir / "tfidf_mat.npy"
        self.cat_path = self.cache_dir / "catalog.json"
        self.catalog: List[Dict[str, Any]] = json.loads(Path(catalog_path).read_text())

        # Build or load TF-IDF
        if force_rebuild or not (self.vec_path.exists() and self.mat_path.exists() and self.cat_path.exists()):
            self._build()
        else:
            self.vectorizer = pickle.loads(self.vec_path.read_bytes())
            self.tfidf = np.load(self.mat_path)
            self.catalog = json.loads(self.cat_path.read_text())

    def _build(self):
        if TfidfVectorizer is None:
            raise RuntimeError("scikit-learn is required for TF-IDF fallback, please ensure it's installed.")
        texts = [_text_blob(it) for it in self.catalog]
        self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1,2))
        self.tfidf = self.vectorizer.fit_transform(texts).astype(np.float32).toarray()
        self.vec_path.write_bytes(pickle.dumps(self.vectorizer))
        np.save(self.mat_path, self.tfidf)
        self.cat_path.write_text(json.dumps(self.catalog))

    def _apply_hard_filters(self, cand_idx: List[int], category: Optional[str], color: Optional[str]) -> List[int]:
        nc = _norm_cat(category)
        ncol = _norm_color(color)
        out = []
        for i in cand_idx:
            it = self.catalog[i]
            it_cat = _norm_cat(it.get("category"))
            it_col = _norm_color(it.get("color"))
            if nc and it_cat != nc:
                continue
            if ncol and it_col != ncol:
                continue
            out.append(i)
        return out

    def search_with_filters(
        self,
        query: str,
        category: Optional[str] = None,
        color: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        top_k: int = 12,
    ) -> List[Dict[str, Any]]:
        if not query:
            query = "popular picks"

        qv = self.vectorizer.transform([query]).astype(np.float32).toarray()[0]
        sims = (self.tfidf @ qv)  # cosine without normalization OK for ranking

        # Start with all items, then HARD filter cat/color/price
        cand = list(range(len(self.catalog)))

        # Price filter first (cheap)
        if min_price is not None or max_price is not None:
            _cand = []
            for i in cand:
                p = float(self.catalog[i].get("price", 0.0))
                if min_price is not None and p < float(min_price): continue
                if max_price is not None and p > float(max_price): continue
                _cand.append(i)
            cand = _cand

        # Strict category + color
        cand = self._apply_hard_filters(cand, category, color)

        # If strict filtering empties the set, relax only color (never category if asked)
        if not cand and category:
            cand = self._apply_hard_filters(list(range(len(self.catalog))), category, None)

        if not cand:
            return []

        # Rank by similarity within candidates
        cand = np.array(cand, dtype=np.int32)
        cand_sims = sims[cand]
        order = np.argsort(-cand_sims)[:top_k]

        out: List[Dict[str, Any]] = []
        for pos in order:
            i = int(cand[int(pos)])
            item = self.catalog[i].copy()
            item["score"] = float(cand_sims[int(pos)])
            out.append(item)
        return out
