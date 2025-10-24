from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io, json, math, re
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests

# ---------- Optional deps (graceful fallbacks) ----------
HAS_TORCH = False
HAS_OPENCLIP = False
try:
    import torch
    HAS_TORCH = True
    torch.set_grad_enabled(False)
except Exception:
    pass

if HAS_TORCH:
    try:
        import open_clip  # type: ignore
        HAS_OPENCLIP = True
    except Exception:
        HAS_OPENCLIP = False

if HAS_TORCH and not HAS_OPENCLIP:
    try:
        from torchvision import models, transforms  # type: ignore
    except Exception:
        HAS_TORCH = False  # effectively disable torch path if torchvision missing

# ---------- Color helpers ----------
import colorsys

NEUTRAL_FIRST = True
COLOR_NAMES = ["black","white","gray","red","orange","yellow","green","blue","purple","brown","assorted"]
COLOR_WORDS = set(COLOR_NAMES) - {"assorted"}

# def _color_from_filename(name: str) -> Optional[str]:
#     n = re.sub(r"[_\\-/]+", " ", name.lower())
#     for c in COLOR_WORDS:
#         if re.search(rf"\b{c}\b", n):
#             return c
#     return None

def _color_from_filename(name: str) -> Optional[str]:
    # normalize separators _, /, \, -  → space
    n = re.sub(r"[\\/_-]+", " ", name.lower())
    for c in COLOR_WORDS:
        if re.search(rf"\b{c}\b", n):
            return c
    return None


def _dominant_color_name(img: Image.Image) -> str:
    """Robust color detector (neutrals first, then hue voting)."""
    try:
        im = img.convert("RGB").resize((160, 160))
        arr = np.asarray(im).astype(np.float32) / 255.0
        h, w, _ = arr.shape
        hsv = np.zeros_like(arr)
        for y in range(h):
            for x in range(w):
                hsv[y, x] = colorsys.rgb_to_hsv(arr[y, x, 0], arr[y, x, 1], arr[y, x, 2])
        H = hsv[:, :, 0] * 360.0
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        mean_s, mean_v = float(S.mean()), float(V.mean())
        if NEUTRAL_FIRST:
            if mean_v < 0.18 and mean_s < 0.25:  # very dark & desat
                return "black"
            if mean_v > 0.90 and mean_s < 0.08:  # very bright & desat
                return "white"
            if mean_s < 0.15:                    # low saturation overall
                return "gray"

        mask = (S >= 0.25) & (V >= 0.25)
        if mask.sum() < (h*w*0.03):
            return "gray" if mean_v > 0.35 else "black"

        Hm = H[mask]
        bins = np.array([0,15,45,75,150,210,270,315,345,360], dtype=np.float32)
        hist, _ = np.histogram(Hm, bins=bins)
        idx = int(np.argmax(hist))
        if idx == 0 or idx >= 7:
            return "red"
        if idx == 1:
            return "orange"
        if idx == 2:
            return "yellow"
        if idx == 3:
            return "green"
        if idx == 4:
            return "blue"
        if idx in (5,6):
            return "purple"
        return "assorted"
    except Exception:
        return "assorted"

def _hsv_hist(img: Image.Image, bins: Tuple[int,int,int]=(12,6,6)) -> np.ndarray:
    im = img.convert("RGB").resize((160,160))
    arr = np.asarray(im).astype(np.float32) / 255.0
    H,S,V = np.zeros_like(arr[:,:,0]), np.zeros_like(arr[:,:,0]), np.zeros_like(arr[:,:,0])
    h, w, _ = arr.shape
    for y in range(h):
        for x in range(w):
            hh, ss, vv = colorsys.rgb_to_hsv(arr[y,x,0], arr[y,x,1], arr[y,x,2])
            H[y,x], S[y,x], V[y,x] = hh, ss, vv

    hb, sb, vb = bins
    Hq = np.clip((H * hb).astype(int), 0, hb-1)
    Sq = np.clip((S * sb).astype(int), 0, sb-1)
    Vq = np.clip((V * vb).astype(int), 0, vb-1)
    hist = np.zeros((hb, sb, vb), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            hist[Hq[i,j], Sq[i,j], Vq[i,j]] += 1.0
    hist = hist.flatten()
    hist /= (hist.sum() + 1e-8)
    return hist.astype(np.float32)

def _hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())

# ---------- Embedding backends ----------
class _OpenClipEncoder:
    def __init__(self):
        model_name, pretrained = "ViT-B-32", "openai"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()

    def encode(self, img: Image.Image) -> np.ndarray:
        import torch  # safe: torch present if we got here
        x = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

class _TorchvisionEncoder:
    def __init__(self):
        from torchvision import models, transforms
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.pre = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def encode(self, img: Image.Image) -> np.ndarray:
        x = self.pre(img).unsqueeze(0)
        with torch.no_grad():
            feats = self.model(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

class _HSVEncoder:
    """Very light fallback; not ideal, but beats random."""
    def __init__(self):
        pass
    def encode(self, img: Image.Image) -> np.ndarray:
        return _hsv_hist(img)  # already L1-normalized

# ---------- Vision Index ----------
class VisionIndex:
    """
    Image index that:
      - Builds an embedding for each catalog image (CLIP -> ResNet -> HSV fallback)
      - Stores HSV histogram and dominant color
      - Reranks by: embedding similarity + color match + histogram intersection + category prior
    """
    def __init__(self, catalog_path: Path, cache_dir: Path, data_dir: Path, force_rebuild: bool=False):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.catalog: List[Dict[str, Any]] = json.loads(Path(catalog_path).read_text())
        self.idxs: List[int] = list(range(len(self.catalog)))

        # Optional overrides: data/overrides.json  ->  {"file.jpg": {"color":"green","category":"shoes"}}
        self.overrides = {}
        ov_path = self.data_dir / "overrides.json"
        if ov_path.exists():
            try:
                self.overrides = json.loads(ov_path.read_text())
            except Exception:
                self.overrides = {}

        # pick encoder
        if HAS_OPENCLIP:
            self.backend = "open_clip"
            self.encoder = _OpenClipEncoder()
        elif HAS_TORCH:
            self.backend = "resnet50"
            self.encoder = _TorchvisionEncoder()
        else:
            self.backend = "hsv"
            self.encoder = _HSVEncoder()

        # cache paths
        self.emb_path = self.cache_dir / f"vision_emb_{self.backend}.npy"
        self.hsv_path = self.cache_dir / "vision_hists.npy"
        self.meta_path = self.cache_dir / "vision_meta.json"

        needs = force_rebuild or not (self.emb_path.exists() and self.hsv_path.exists() and self.meta_path.exists())
        if needs:
            self._rebuild()
        else:
            self.embs = np.load(self.emb_path)
            self.hists = np.load(self.hsv_path)
            self.meta = json.loads(self.meta_path.read_text())

    def _load_image(self, rel_path: str) -> Optional[Image.Image]:
        fp = self.data_dir / rel_path
        try:
            return Image.open(fp).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            return None

    def _apply_overrides(self, filename: str, color: str, category: str) -> Tuple[str,str]:
        ov = self.overrides.get(filename)
        if isinstance(ov, dict):
            color = ov.get("color", color) or color
            category = ov.get("category", category) or category
        return color, category

    def _rebuild(self):
        embs: List[np.ndarray] = []
        hists: List[np.ndarray] = []
        meta: List[Dict[str, Any]] = []

        for i, item in enumerate(self.catalog):
            rel = item.get("image_path")
            if not rel:
                continue
            img = self._load_image(rel)
            if img is None:
                continue

            # embedding + histogram
            emb = self.encoder.encode(img)
            hist = _hsv_hist(img)

            # color & category (with override + filename hint)
            file_color = _color_from_filename(Path(rel).stem)
            dom_color = _dominant_color_name(img)
            color = file_color or dom_color or item.get("color", "assorted")
            category = item.get("category", "assorted")
            color, category = self._apply_overrides(Path(rel).name, color, category)

            embs.append(emb)
            hists.append(hist)
            meta.append({
                "idx": i,
                "id": item.get("id"),
                "category": category,
                "color": color,
                "image_path": rel,
            })

        # pad-consistent array
        self.embs = np.asarray(embs, dtype=np.float32)
        # L2 normalize embeddings if not already (hsv hist is L1, handled in comparator)
        if self.backend in ("open_clip", "resnet50"):
            norms = np.linalg.norm(self.embs, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            self.embs = (self.embs / norms).astype(np.float32)

        self.hists = np.asarray(hists, dtype=np.float32)
        self.meta = meta

        np.save(self.emb_path, self.embs)
        np.save(self.hsv_path, self.hists)
        Path(self.meta_path).write_text(json.dumps(self.meta))

    # ---------- Scoring ----------
    def _embed(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray, str]:
        emb = self.encoder.encode(img)
        if self.backend in ("open_clip","resnet50"):
            n = np.linalg.norm(emb)
            if n > 0: emb = (emb / n).astype(np.float32)
        hist = _hsv_hist(img)
        q_color = _dominant_color_name(img)
        return emb.astype(np.float32), hist.astype(np.float32), q_color

    def _score(self, q_emb: np.ndarray, q_hist: np.ndarray, q_color: str) -> np.ndarray:
        # base similarity
        if self.backend in ("open_clip","resnet50"):
            base = (self.embs @ q_emb)  # cosine (both L2)
        else:
            base = np.array([_hist_intersection(h, q_hist) for h in self.hists], dtype=np.float32)

        # color bonus
        color_bonus = np.zeros(len(self.meta), dtype=np.float32)
        for i, m in enumerate(self.meta):
            if q_color != "assorted" and m["color"] == q_color:
                color_bonus[i] = 0.12

        # histogram similarity (helps even with CLIP)
        hist_sim = np.array([_hist_intersection(h, q_hist) for h in self.hists], dtype=np.float32) * 0.25

        score = base + color_bonus + hist_sim
        return score.astype(np.float32)

    def _category_prior(self, scores: np.ndarray, top_m: int = 40) -> Dict[str, float]:
        """Estimate the best category for the query from the top-M candidates and return a boost per category."""
        order = np.argsort(-scores)[:min(top_m, len(scores))]
        counts: Dict[str, int] = {}
        for j in order:
            c = self.meta[int(j)]["category"]
            counts[c] = counts.get(c, 0) + 1
        if not counts:
            return {}
        # normalize to [0, 0.15]
        total = sum(counts.values())
        priors = {c: (cnt/total)*0.15 for c, cnt in counts.items()}
        # strongest category gets a tiny extra nudge
        best_c = max(priors, key=priors.get)
        priors[best_c] += 0.05
        return priors

    # ---------- Public API ----------
    def search_image_path(self, path: Path, top_k: int = 8) -> List[Dict[str, Any]]:
        img = Image.open(path).convert("RGB")
        return self._search_image(img, filename_hint=path.name, top_k=top_k)

    def search_image_url(self, url: str, top_k: int = 8) -> List[Dict[str, Any]]:
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return []
        return self._search_image(img, filename_hint=Path(url).name, top_k=top_k)

    def _search_image(self, img: Image.Image, filename_hint: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        q_emb, q_hist, q_color = self._embed(img)
        base_scores = self._score(q_emb, q_hist, q_color)

        # Category prior (choose category from visual neighbors)
        cat_boost_map = self._category_prior(base_scores, top_m=40)
        cat_boost = np.zeros_like(base_scores)
        for i, m in enumerate(self.meta):
            cat_boost[i] = cat_boost_map.get(m["category"], 0.0)

        final = base_scores + cat_boost

        order = np.argsort(-final)[:min(top_k*3, len(final))]  # take a bit more, then filter by strict rules
        results: List[Tuple[int, float]] = [(int(j), float(final[int(j)])) for j in order]

        # Assemble items; keep category consistency by preferring the dominant category
        # (already encouraged by cat_boost), but we won’t *hard* filter — we just rank.
        out: List[Dict[str, Any]] = []
        for j, sc in results[:top_k]:
            m = self.meta[j]
            it = self.catalog[m["idx"]].copy()
            it["score"] = round(sc, 6)
            # Harmonize color if catalog color is unknown/assorted but we detected a good color from query
            if it.get("color") in (None, "", "assorted") and q_color not in ("assorted", None):
                it["color"] = q_color
                it["title"] = re.sub(r"^[A-Za-z]+\s+", f"{q_color.capitalize()} ", it.get("title",""), count=1) or it.get("title","")
            out.append(it)

        return out[:top_k]
