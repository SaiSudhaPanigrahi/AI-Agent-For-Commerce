from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import io, json, requests, colorsys
import numpy as np
from PIL import Image

def _hsv_hist(img: Image.Image, bins_h=24,bins_s=8,bins_v=8) -> np.ndarray:
    img = img.convert("RGB").resize((256,256))
    arr = np.asarray(img).astype(np.float32)/255.0
    h,w,_ = arr.shape
    H = np.zeros((h,w), dtype=np.int32)
    S = np.zeros((h,w), dtype=np.int32)
    V = np.zeros((h,w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            hh,ss,vv = colorsys.rgb_to_hsv(arr[y,x,0],arr[y,x,1],arr[y,x,2])
            H[y,x] = min(bins_h-1, int(hh*bins_h))
            S[y,x] = min(bins_s-1, int(ss*bins_s))
            V[y,x] = min(bins_v-1, int(vv*bins_v))
    hist = np.zeros((bins_h,bins_s,bins_v), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            hist[H[y,x], S[y,x], V[y,x]] += 1.0
    hist /= max(1.0, hist.sum())
    return hist.flatten()

def _dominant_color_name(img: Image.Image) -> str:
    im = img.convert("RGB").resize((64,64))
    arr = np.asarray(im).astype(np.float32)/255.0
    s=v=0.0; H=0.0
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            hh,ss,vv = colorsys.rgb_to_hsv(arr[y,x,0],arr[y,x,1],arr[y,x,2])
            H += hh; s += ss; v += vv
    H = (H/(arr.shape[0]*arr.shape[1]))*360
    s /= (arr.shape[0]*arr.shape[1]); v /= (arr.shape[0]*arr.shape[1])
    if s < 0.1 and v > 0.9: return "white"
    if s < 0.12 and v < 0.25: return "black"
    if s < 0.15: return "gray"
    if   345<=H or H<15: return "red"
    elif 15<=H<45: return "orange"
    elif 45<=H<75: return "yellow"
    elif 75<=H<165: return "green"
    elif 165<=H<255: return "blue"
    elif 255<=H<315: return "purple"
    elif 315<=H<345: return "red"
    return "assorted"

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

class VisionIndex:
    def __init__(self, catalog_path: Path, cache_dir: Path, data_dir: Path, force_rebuild: bool=False):
        self.catalog = json.loads(Path(catalog_path).read_text())
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.vecs_path = self.cache_dir / "vision_vecs.npy"
        self.idx_path  = self.cache_dir / "vision_idx.json"

        if force_rebuild or not (self.vecs_path.exists() and self.idx_path.exists()):
            self._build()
        else:
            self.vecs = np.load(self.vecs_path)
            self.idx_meta = json.loads(self.idx_path.read_text())

    def _build(self):
        vecs = []; meta = []
        for i, it in enumerate(self.catalog):
            img_path = self.data_dir / it["image_path"]
            if not img_path.exists(): continue
            try:
                img = Image.open(img_path)
                vec = _hsv_hist(img)
            except Exception:
                continue
            vecs.append(vec)
            meta.append({"i": i, "color": it.get("color","assorted").lower()})
        self.vecs = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0,1), dtype=np.float32)
        self.idx_meta = meta
        np.save(self.vecs_path, self.vecs)
        self.idx_path.write_text(json.dumps(self.idx_meta))

    def _rank(self, qv: np.ndarray, q_color: Optional[str], top_k: int) -> list[dict]:
        if self.vecs.shape[0]==0: return []
        sims = np.array([_cosine(qv, v) for v in self.vecs], dtype=np.float32)
        boosts = np.zeros_like(sims)
        if q_color:
            for j,m in enumerate(self.idx_meta):
                col = (m.get("color") or "assorted").lower()
                if col == q_color: boosts[j]+=0.12
                elif col in ("black","white","gray"): boosts[j]+=0.02
        score = sims + boosts
        order = np.argsort(-score)[:max(1, top_k)]
        out=[]
        for pos in order:
            meta = self.idx_meta[int(pos)]
            item = self.catalog[int(meta["i"])].copy()
            item["score"] = float(score[int(pos)])
            out.append(item)
        return out

    def search_image_path(self, image_path: Path, top_k: int=8) -> list[dict]:
        img = Image.open(image_path)
        qv = _hsv_hist(img)
        qc = _dominant_color_name(img)
        return self._rank(qv, qc, top_k)

    def search_image_url(self, url: str, top_k: int=8) -> list[dict]:
        r = requests.get(url, timeout=15); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        qv = _hsv_hist(img); qc = _dominant_color_name(img)
        return self._rank(qv, qc, top_k)
