import io
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _safe_get(url: str, timeout: float = 7.0) -> Optional[bytes]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Mercury/1.0) image-fetch",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


class ImageSearch:
    """
    CLIP image->catalog visual search with:
      - robust fetching
      - lazy cached catalog embeddings (image first, text fallback)
      - similarity thresholding and category-aware rerank
    """

    def __init__(self, products: List[Dict[str, Any]]):
        self.products = products
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._cache_img_emb: Dict[str, torch.Tensor] = {}
        self._cache_txt_emb: Dict[str, torch.Tensor] = {}

    # ---------- embeddings ----------
    def _embed_image_bytes(self, content: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return _norm(feats.squeeze(0)).cpu()

    def _embed_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return _norm(feats.squeeze(0)).cpu()

    def _product_text(self, p: Dict[str, Any]) -> str:
        parts = [p.get("title", ""), p.get("brand", ""), p.get("category", ""), p.get("description", "")]
        if p.get("tags"):
            parts.append(" ".join(p["tags"]))
        return " | ".join([t for t in parts if t])

    def _catalog_vector(self, p: Dict[str, Any]) -> Optional[torch.Tensor]:
        pid = p.get("id") or p.get("title")
        if pid in self._cache_img_emb:
            return self._cache_img_emb[pid]
        # try image
        url = (p.get("image_url") or "").strip()
        if url:
            content = _safe_get(url)
            if content:
                try:
                    vec = self._embed_image_bytes(content)
                    self._cache_img_emb[pid] = vec
                    return vec
                except Exception:
                    pass
        # fallback text
        if pid not in self._cache_txt_emb:
            self._cache_txt_emb[pid] = self._embed_text(self._product_text(p))
        return self._cache_txt_emb[pid]

    # ---------- search ----------
    def search_by_image_url(self, image_url: str, top_k: int = 8) -> List[Dict[str, Any]]:
        content = _safe_get(image_url)
        if not content:
            # try treating the URL string as text query fallback
            qvec = self._embed_text(image_url)
        else:
            try:
                qvec = self._embed_image_bytes(content)
            except Exception:
                qvec = self._embed_text(image_url)

        # build catalog matrix lazily
        vecs: List[Tuple[int, torch.Tensor]] = []
        for i, p in enumerate(self.products):
            v = self._catalog_vector(p)
            if v is not None and torch.isfinite(v).all():
                vecs.append((i, v))

        if not vecs:
            return []

        cat = torch.stack([v for _, v in vecs], dim=0)  # [N, D]
        sims = (cat @ qvec)  # cosine due to normalization
        scores = sims.numpy().tolist()

        # basic top-k
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        order = order[: max(top_k * 4, 16)]  # expand pool for thresholding/rerank

        # dynamic threshold: keep reasonably similar results only
        max_s = max(scores[i] for i in order) if order else 0.0
        thr = max(0.22, 0.72 * float(max_s))  # adaptive + base floor
        keep = [i for i in order if scores[i] >= thr]

        # category-aware rerank: boost items from the dominant category among top few
        if keep:
            top_cat_counts: Dict[str, int] = {}
            for idx in keep[:8]:
                cat_name = str(self.products[vecs[idx][0]].get("category", "")).lower()
                top_cat_counts[cat_name] = top_cat_counts.get(cat_name, 0) + 1
            dom_cat = max(top_cat_counts.items(), key=lambda kv: kv[1])[0] if top_cat_counts else ""
            def boost(idx: int) -> float:
                base = scores[idx]
                p = self.products[vecs[idx][0]]
                return base + (0.05 if str(p.get("category","")).lower() == dom_cat and base > thr else 0.0)
            keep = sorted(keep, key=boost, reverse=True)

        # map back to products
        out: List[Dict[str, Any]] = []
        for idx in keep[:top_k]:
            prod_idx = vecs[idx][0]
            out.append(self.products[prod_idx])

        return out
