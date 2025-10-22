from pathlib import Path
from typing import List, Dict, Any
import io
import json
import numpy as np
from PIL import Image
import requests

def _histogram_vector(img: Image.Image, bins: int=64) -> np.ndarray:
    img = img.convert("RGB").resize((256,256))
    arr = np.array(img)
    vecs = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:,:,ch], bins=bins, range=(0,255), density=True)
        vecs.append(hist.astype(np.float32))
    v = np.concatenate(vecs, axis=0)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def _cosine(a,b):
    return float(np.dot(a,b) / ((np.linalg.norm(a)+1e-8)*(np.linalg.norm(b)+1e-8)))

class VisionIndex:
    def __init__(self, catalog_path: Path, cache_dir: Path, data_dir: Path, force_rebuild: bool=False):
        self.catalog_path = catalog_path
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.vec_path = cache_dir / "image_histograms.npy"
        with catalog_path.open() as f:
            self.catalog = json.load(f)
        if force_rebuild or not self.vec_path.exists():
            self._rebuild()
        else:
            self._load()

    def _rebuild(self):
        vecs = []
        for it in self.catalog:
            p = self.data_dir / it["image_path"]
            try:
                img = Image.open(p)
                vecs.append(_histogram_vector(img))
            except Exception:
                vecs.append(np.zeros(64*3, dtype=np.float32))
        X = np.stack(vecs, axis=0) if len(vecs) else np.zeros((0,64*3), dtype=np.float32)
        np.save(self.vec_path, X)
        self._X = X

    def _load(self):
        self._X = np.load(self.vec_path, allow_pickle=False)

    def _search_by_vec(self, qv: np.ndarray, top_k: int=8) -> List[Dict[str, Any]]:
        if len(self.catalog) == 0:
            return []
        sims = np.array([_cosine(qv, v) for v in self._X]) if len(self._X) else np.array([])
        idx = np.argsort(-sims)[:top_k]
        out = []
        for i in idx:
            item = self.catalog[i].copy()
            item["score"] = float(sims[i])
            out.append(item)
        return out

    def search_image_path(self, image_path: Path, top_k: int=8) -> List[Dict[str, Any]]:
        img = Image.open(image_path)
        qv = _histogram_vector(img)
        return self._search_by_vec(qv, top_k=top_k)

    def search_image_url(self, url: str, top_k: int=8) -> List[Dict[str, Any]]:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        qv = _histogram_vector(img)
        return self._search_by_vec(qv, top_k=top_k)
