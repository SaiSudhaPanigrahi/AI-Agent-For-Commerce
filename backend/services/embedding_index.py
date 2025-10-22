from pathlib import Path
import os, json, time, pickle, re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer

# Primary (online) model; SDK expects "models/..." names.
EMBED_MODEL = "models/text-embedding-004"

CATS = {
    "bag": "bags", "bags": "bags",
    "cap": "caps", "caps": "caps",
    "jacket": "jackets", "jackets": "jackets",
    "shoe": "shoes", "shoes": "shoes",
}

PRICE_PATTERNS = {
    "under": re.compile(r"(?:under|below|less than|<=|≤)\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    "over":  re.compile(r"(?:over|above|more than|>=|≥)\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    "between": re.compile(r"(?:between|from)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
}

def _cosine_sim(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))

def _extract_vec(res) -> np.ndarray:
    """
    Handle multiple google-generativeai return shapes:
    - obj.embedding.values
    - obj.embeddings[0].values
    - {"embedding": {"values": [...]}}
    - {"embeddings": [{"values": [...]}]}
    - sometimes 'value' instead of 'values'
    """
    # attribute-style
    if hasattr(res, "embedding") and hasattr(res.embedding, "values"):
        return np.array(res.embedding.values, dtype=np.float32)
    if hasattr(res, "embeddings") and res.embeddings:
        first = res.embeddings[0]
        vals = getattr(first, "values", None)
        if vals is not None:
            return np.array(vals, dtype=np.float32)
    # dict-style
    if isinstance(res, dict):
        emb = res.get("embedding")
        if isinstance(emb, dict):
            vals = emb.get("values") or emb.get("value")
            if vals is not None:
                return np.array(vals, dtype=np.float32)
        embs = res.get("embeddings")
        if isinstance(embs, list) and embs:
            vals = embs[0].get("values") or embs[0].get("value")
            if vals is not None:
                return np.array(vals, dtype=np.float32)
    raise RuntimeError("Unexpected embedding response format from Gemini.")

def parse_query_constraints(q: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Returns (category, min_price, max_price).
    - category mapped to plural ('bags','caps','jackets','shoes') if detected
    - prices are floats; None if not present
    Rules:
      - "under/below/less than/<=/≤ N"  => max=N  (we treat 'under 100' as <= 100)
      - "over/above/more than/>=/≥ N"   => min=N
      - "between/from X and/to Y"       => min=X, max=Y (order-insensitive)
    """
    q_low = q.lower()

    # category
    cat_detected = None
    for key, canon in CATS.items():
        if re.search(rf"\b{re.escape(key)}\b", q_low):
            cat_detected = canon
            break

    # price
    min_p, max_p = None, None

    m_between = PRICE_PATTERNS["between"].search(q_low)
    if m_between:
        a = float(m_between.group(1))
        b = float(m_between.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        min_p, max_p = lo, hi
        return cat_detected, min_p, max_p

    m_under = PRICE_PATTERNS["under"].search(q_low)
    if m_under:
        max_p = float(m_under.group(1))

    m_over = PRICE_PATTERNS["over"].search(q_low)
    if m_over:
        min_p = float(m_over.group(1))

    return cat_detected, min_p, max_p

class EmbeddingIndex:
    """
    Text RAG with resilient embedding:
      1) Try Gemini embeddings.
      2) On any failure, transparently fall back to offline TF-IDF (no API).
    Caches artifacts under backend/.cache/.
    Adds query-time filtering on category and price constraints.
    """
    def __init__(self, catalog_path: Path, cache_dir: Path, force_rebuild: bool=False):
        self.catalog_path = catalog_path
        self.cache_dir = cache_dir
        self.vec_path = cache_dir / "text_embeddings.npy"   # for Gemini vectors
        self.meta_path = cache_dir / "text_meta.json"
        self.mode_path = cache_dir / "text_mode.txt"        # "gemini" or "tfidf"
        self.tfidf_model_path = cache_dir / "tfidf_vectorizer.pkl"
        self.tfidf_matrix_path = cache_dir / "tfidf_matrix.npy"

        with catalog_path.open() as f:
            self.catalog = json.load(f)

        # Configure Gemini if key exists; we'll still fall back if anything fails.
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

        if force_rebuild or (not self.meta_path.exists() or not self.mode_path.exists()):
            self._rebuild_resilient()
        else:
            self._load()

    # ---------- Embedding helpers ----------
    def _embed_one_gemini(self, text: str, retries: int = 3, backoff: float = 0.8) -> np.ndarray:
        last_err = None
        for i in range(retries):
            try:
                res = genai.embed_content(
                    model=EMBED_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                return _extract_vec(res)
            except Exception as e:
                last_err = e
                time.sleep(backoff * (2**i))
        raise last_err

    def _embed_batch_gemini(self, texts: List[str]) -> np.ndarray:
        vecs = [self._embed_one_gemini(t) for t in texts]
        return np.stack(vecs, axis=0) if vecs else np.zeros((0, 768), dtype=np.float32)

    # ---------- Build paths ----------
    def _product_texts(self) -> List[str]:
        texts = []
        for it in self.catalog:
            blob = f"{it['title']} | {it['description']} | {it['category']} | {it['color']} | tags: {', '.join(it.get('tags', []))}"
            texts.append(blob)
        return texts

    # ---------- Rebuild with resilience ----------
    def _rebuild_resilient(self):
        texts = self._product_texts()

        # If nothing in catalog, write empty caches.
        if not texts:
            np.save(self.vec_path, np.zeros((0, 768), dtype=np.float32))
            with self.meta_path.open("w") as f:
                json.dump({"ids":[it["id"] for it in self.catalog]}, f)
            self._X = np.zeros((0, 768), dtype=np.float32)
            self.mode_path.write_text("gemini")
            return

        # Try Gemini if we have an API key
        if self.api_key:
            try:
                X = self._embed_batch_gemini(texts)
                np.save(self.vec_path, X)
                with self.meta_path.open("w") as f:
                    json.dump({"ids":[it["id"] for it in self.catalog]}, f)
                self._X = X
                self.mode_path.write_text("gemini")
                return
            except Exception:
                # fall back to TF-IDF below
                pass

        # Fallback: TF-IDF offline (no network)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1,2),
            min_df=1,
            max_df=0.95
        )
        tfidf = vectorizer.fit_transform(texts)  # (N, V) sparse
        X = tfidf.toarray().astype(np.float32)
        np.save(self.tfidf_matrix_path, X)
        with open(self.tfidf_model_path, "wb") as f:
            pickle.dump(vectorizer, f)
        with self.meta_path.open("w") as f:
            json.dump({"ids":[it["id"] for it in self.catalog]}, f)
        self._X = X
        self.mode_path.write_text("tfidf")

    def _load(self):
        mode = self.mode_path.read_text().strip()
        with self.meta_path.open() as f:
            self._meta = json.load(f)
        if mode == "gemini":
            self._X = np.load(self.vec_path, allow_pickle=False)
        elif mode == "tfidf":
            self._X = np.load(self.tfidf_matrix_path, allow_pickle=False)
            with open(self.tfidf_model_path, "rb") as f:
                self._tfidf_vectorizer = pickle.load(f)
        else:
            # unknown -> force rebuild
            self._rebuild_resilient()
            mode = self.mode_path.read_text().strip()
        self._mode = mode

    # ---------- Search with constraints ----------
    def _filter_indices(self, query: str) -> List[int]:
        cat, min_p, max_p = parse_query_constraints(query)
        idxs = list(range(len(self.catalog)))
        if cat:
            idxs = [i for i in idxs if self.catalog[i].get("category") == cat]
        if min_p is not None:
            idxs = [i for i in idxs if float(self.catalog[i].get("price", 0.0)) >= min_p]
        if max_p is not None:
            # treat "under 100" as <= 100
            idxs = [i for i in idxs if float(self.catalog[i].get("price", 0.0)) <= max_p]
        return idxs

    def search(self, query: str, top_k: int=8) -> List[Dict[str, Any]]:
        if len(self.catalog) == 0:
            return []

        # choose candidate subset by constraints
        cand_idxs = self._filter_indices(query)
        if not cand_idxs:
            # no matches under constraints → fallback to full set
            cand_idxs = list(range(len(self.catalog)))

        mode = getattr(self, "_mode", "gemini")

        if mode == "gemini":
            # Online embeddings for the query (with fallback if key disappeared)
            try:
                qv = self._embed_batch_gemini([query])[0]
            except Exception:
                # fall back on-the-fly: convert corpus to TF-IDF once
                self._rebuild_resilient()
                return self.search(query, top_k)
            sims_all = np.array([_cosine_sim(qv, v) for v in self._X]) if len(self._X) else np.array([])
        else:  # tfidf
            if not hasattr(self, "_tfidf_vectorizer"):
                with open(self.tfidf_model_path, "rb") as f:
                    self._tfidf_vectorizer = pickle.load(f)
            qvec = self._tfidf_vectorizer.transform([query]).toarray().astype(np.float32)[0]
            denom = (np.linalg.norm(self._X, axis=1) + 1e-8) * (np.linalg.norm(qvec) + 1e-8)
            sims_all = (self._X @ qvec) / denom

        # restrict to candidate indices and rank
        sims = np.array([sims_all[i] for i in cand_idxs], dtype=np.float32)
        order = np.argsort(-sims)[:top_k]
        out = []
        for pos in order:
            i = cand_idxs[int(pos)]
            item = self.catalog[i].copy()
            item["score"] = float(sims[int(pos)])
            out.append(item)

        return out

    def search_with_filters(self, query: str, filters: Dict[str, Any], top_k: int = 8):
        """
        Same as search(), but uses explicit filters (from Gemini) instead of parsing the query.
        Supported keys in filters: category, min_price, max_price, color
        """
        # Build candidate set using explicit filters
        idxs = list(range(len(self.catalog)))
        cat = filters.get("category")
        mn = filters.get("min_price")
        mx = filters.get("max_price")
        col = filters.get("color")
        if cat:
            idxs = [i for i in idxs if self.catalog[i].get("category") == cat]
        if mn is not None:
            idxs = [i for i in idxs if float(self.catalog[i].get("price", 0.0)) >= float(mn)]
        if mx is not None:
            idxs = [i for i in idxs if float(self.catalog[i].get("price", 0.0)) <= float(mx)]

        # strict color if possible; otherwise keep idxs and boost later
        strict = None
        if col:
            strict = []
            for i in idxs:
                it = self.catalog[i]
                blob = f"{it.get('title', '')} {it.get('description', '')}".lower()
                clr = (it.get("color", "") or "").lower().replace("gray", "grey")
                if (col in blob) or (col in clr):
                    strict.append(i)
        cand = strict if (strict and len(strict) > 0) else idxs
        if not cand:
            cand = list(range(len(self.catalog)))

        # rank like search()
        mode = getattr(self, "_mode", "gemini")
        if mode == "gemini":
            try:
                qv = self._embed_batch_gemini([query])[0]
            except Exception:
                self._rebuild_resilient()
                return self.search_with_filters(query, filters, top_k)
            sims_all = np.array([_cosine_sim(qv, v) for v in self._X]) if len(self._X) else np.array([])
        else:
            if not hasattr(self, "_tfidf_vectorizer"):
                with open(self.tfidf_model_path, "rb") as f:
                    self._tfidf_vectorizer = pickle.load(f)
            qvec = self._tfidf_vectorizer.transform([query]).toarray().astype(np.float32)[0]
            denom = (np.linalg.norm(self._X, axis=1) + 1e-8) * (np.linalg.norm(qvec) + 1e-8)
            sims_all = (self._X @ qvec) / denom

        sims = np.array([sims_all[i] for i in cand], dtype=np.float32)

        # small color boost
        if col:
            boosts = []
            for i in cand:
                it = self.catalog[i]
                blob = f"{it.get('title', '')} {it.get('description', '')}".lower()
                clr = (it.get("color", "") or "").lower().replace("gray", "grey")
                boosts.append(0.25 if (col in blob) or (col in clr) else 0.0)
            sims = sims + np.array(boosts, dtype=np.float32)

        intent_terms = []
        ql = query.lower()
        for t in ["running", "trail", "hiking", "gym", "workout", "sport", "sporty", "formal", "casual",
                  "winter", "summer", "waterproof", "leather", "canvas", "breathable",
                  "lightweight", "women", "ladies", "men", "unisex"]:
            if t in ql:
                intent_terms.append(t)
        if intent_terms:
            _tag_boosts = []
            for i in cand:
                tags = [str(x).lower() for x in self.catalog[i].get("tags", [])]
                hit = any(term in tags for term in intent_terms)
                _tag_boosts.append(0.18 if hit else 0.0)
            sims = sims + np.array(_tag_boosts, dtype=np.float32)

        order = np.argsort(-sims)[:top_k]
        out = []
        for pos in order:
            i = cand[int(pos)]
            item = self.catalog[i].copy()
            item["score"] = float(sims[int(pos)])
            out.append(item)
        return out
