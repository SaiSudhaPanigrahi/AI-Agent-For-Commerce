from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json, re
import numpy as np

# Prefer Sentence-Transformers; fall back to TF-IDF if not installed
USE_ST = True
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    USE_ST = False

if not USE_ST:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

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
    tags = " ".join(it.get("tags", []))
    return " ".join(str(x) for x in [
        it.get("title",""), it.get("description",""),
        it.get("category",""), it.get("color",""),
        tags
    ] if x)

class TextIndex:
    """
    RAG vector store + strict filters + light re-ranking.
    Ranking signals (after filters):
      - semantic similarity (embedding cosine)
      - + purpose/tag overlap boost
      - + price proximity boost (if max_price set), never breaks the ≤ max rule
    """
    def __init__(self, catalog_path: Path, cache_dir: Path, force_rebuild: bool=False):
        self.catalog: List[Dict[str,Any]] = json.loads(Path(catalog_path).read_text())
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if USE_ST:
            self.vec_path = self.cache_dir / "sent_vecs.npy"
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
            if force_rebuild or not self.vec_path.exists():
                texts = [_text_blob(it) for it in self.catalog]
                embs = self.model.encode(texts, normalize_embeddings=True)
                self.vecs = np.asarray(embs, dtype=np.float32)
                np.save(self.vec_path, self.vecs)
            else:
                self.vecs = np.load(self.vec_path)
            self._encode_query = lambda q: self.model.encode([q or "popular picks"], normalize_embeddings=True)[0].astype(np.float32)
        else:
            self.vec_path = self.cache_dir / "tfidf_mat.npy"
            self.vocab_path = self.cache_dir / "tfidf_vocab.npy"
            self.idf_path = self.cache_dir / "tfidf_idf.npy"
            texts = [_text_blob(it) for it in self.catalog]
            rebuild = not (self.vec_path.exists() and self.vocab_path.exists() and self.idf_path.exists()) or force_rebuild
            if rebuild:
                self._tfidf = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1,2))
                X = self._tfidf.fit_transform(texts)
                mat = X.astype(np.float32).toarray()
                np.save(self.vec_path, mat)
                np.save(self.vocab_path, np.array(self._tfidf.get_feature_names_out(), dtype=object))
                np.save(self.idf_path, self._tfidf.idf_.astype(np.float32))
            else:
                vocab = np.load(self.vocab_path, allow_pickle=True)
                idf = np.load(self.idf_path)
                self._tfidf = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1,2), vocabulary=list(vocab))
                try:
                    self._tfidf._tfidf._idf_diag = None
                    self._tfidf.idf_ = idf
                except Exception:
                    pass
                mat = np.load(self.vec_path)
            norms = np.linalg.norm(mat, axis=1, keepdims=True); norms[norms==0]=1.0
            self.vecs = (mat / norms).astype(np.float32)
            def _enc(q: str) -> np.ndarray:
                Xq = self._tfidf.transform([q or "popular picks"]).astype(np.float32).toarray()[0]
                n = np.linalg.norm(Xq)
                return (Xq / n).astype(np.float32) if n>0 else Xq
            self._encode_query = _enc

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
            if (min_price is not None) and (p < float(min_price)):
                continue
            if (max_price is not None) and (p > float(max_price)):  # STRICT ≤ max
                continue
            out.append(i)
        return out

    def search_with_filters(self, query: str,
                            category: Optional[str]=None, color: Optional[str]=None,
                            min_price: Optional[float]=None, max_price: Optional[float]=None,
                            top_k: int=12) -> List[Dict[str, Any]]:
        q = (query or "popular picks")
        qv = self._encode_query(q)
        sims = (self.vecs @ qv)

        all_idx = list(range(len(self.catalog)))
        idx = self._apply_filters(all_idx, category, color, min_price, max_price)
        if not idx and category:
            idx = self._apply_filters(all_idx, category, None, min_price, max_price)
        if not idx:
            return []

        # Light re-ranking
        q_words = set(w for w in re.findall(r"[a-zA-Z]+", q.lower()))
        def tag_boost(it):
            tags = [t.lower() for t in it.get("tags", [])]
            overlap = len(q_words.intersection(tags))
            return 0.12 * min(overlap, 2)

        def price_boost(it):
            if max_price is None:
                return 0.0
            p = float(it.get("price", 0.0))
            if p > float(max_price):
                return -1.0  # should be filtered already; safety
            # closer to max gets small boost (value-for-budget)
            return 0.10 * (p / float(max_price))

        cand = np.array(idx, dtype=np.int32)
        base = sims[cand]
        extra = np.array([tag_boost(self.catalog[i]) + price_boost(self.catalog[i]) for i in cand], dtype=np.float32)
        score = base + extra

        order = np.argsort(-score)[:top_k]
        out: List[Dict[str,Any]] = []
        for j in order:
            i = int(cand[int(j)])
            item = self.catalog[i].copy()
            item["score"] = float(score[int(j)])
            out.append(item)
        return out
