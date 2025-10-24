from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re, json, colorsys, math, hashlib
import numpy as np
from PIL import Image

# ---- Supported categories (same as UI expects) ----
CAT_MAP = {
    "bag":"bags","bags":"bags",
    "shoe":"shoes","shoes":"shoes",
    "jacket":"jackets","jackets":"jackets",
    "cap":"caps","caps":"caps",
}

# Canonical color names
COLOR_NAMES = ["black","white","gray","red","orange","yellow","green","blue","purple","brown","assorted"]

# Quick filename -> color override
COLOR_WORDS = set(COLOR_NAMES) - {"assorted"}

def _color_from_filename(name: str) -> str | None:
    n = re.sub(r"[_\-]+"," ", name.lower())
    for c in COLOR_WORDS:
        if re.search(rf"\b{c}\b", n):
            return c
    return None

# ---- Robust HSV color detection ----
def _dominant_color_name(img_path: Path) -> str:
    """
    Detect dominant color with a bias toward neutral colors when saturation is low.
    Steps:
      1) Downsample and convert to HSV
      2) Handle neutral colors FIRST (black/white/gray) by thresholds
      3) Build hue histogram on pixels with enough saturation/brightness
      4) Map hue peak to a canonical color
    """
    try:
        im = Image.open(img_path).convert("RGB").resize((160, 160))
        arr = np.asarray(im).astype(np.float32) / 255.0
        h, w, _ = arr.shape
        hsv = np.zeros_like(arr)
        for y in range(h):
            for x in range(w):
                hsv[y, x] = colorsys.rgb_to_hsv(arr[y, x, 0], arr[y, x, 1], arr[y, x, 2])
        H = hsv[:, :, 0] * 360.0
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        # Neutral checks — do these BEFORE hue mapping
        mean_s, mean_v = float(S.mean()), float(V.mean())
        # Very dark -> black
        if mean_v < 0.18 and mean_s < 0.25:
            return "black"
        # Very bright & desaturated -> white
        if mean_v > 0.9 and mean_s < 0.08:
            return "white"
        # Low saturation overall -> gray
        if mean_s < 0.15:
            return "gray"

        # Keep only sufficiently saturated/bright pixels for hue voting
        mask = (S >= 0.25) & (V >= 0.25)
        if mask.sum() < (h*w*0.03):
            # Not enough colored pixels -> fall back to neutral heuristic
            return "gray" if mean_v > 0.35 else "black"

        Hm = H[mask]
        # 360-degree histogram (coarse -> fewer mislabels)
        bins = np.array([0,15,45,75,150,210,270,315,345,360], dtype=np.float32)
        hist, _ = np.histogram(Hm, bins=bins)
        idx = int(np.argmax(hist))
        # Map bin index to color
        # bins: [0,15)=red, [15,45)=orange, [45,75)=yellow, [75,150)=green,
        # [150,210)=blue, [210,270)=purple, [270,315)=purple->red bridge,
        # [315,345)=red, [345,360)=red
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
        if idx in (5, 6):
            return "purple"
        return "assorted"
    except Exception:
        return "assorted"

# ---- Varied, realistic content generation (deterministic) ----

# Per-category title variants + purpose tags
SHOE_TYPES = [
    ("Running Shoes", ["running","road","cushion"]),
    ("Trail Runners", ["trail","outdoor","grip"]),
    ("Training Sneakers", ["training","gym","cross"]),
    ("Casual Sneakers", ["casual","everyday"]),
]
BAG_TYPES = [
    ("Everyday Tote", ["tote","commute","everyday"]),
    ("Travel Duffel", ["travel","duffel","weekend"]),
    ("Daypack Backpack", ["backpack","daypack","school"]),
    ("Crossbody Bag", ["crossbody","compact","city"]),
]
JACKET_TYPES = [
    ("Insulated Puffer Jacket", ["winter","insulated","puffer"]),
    ("Lightweight Windbreaker", ["windbreaker","windproof"]),
    ("Rain Shell Jacket", ["rain","waterproof"]),
    ("Fleece Zip Jacket", ["fleece","cozy"]),
]
CAP_TYPES = [
    ("Thermal Knit Beanie", ["winter","thermal","beanie"]),
    ("Baseball Cap", ["classic","sun"]),
    ("Running Cap", ["running","lightweight","breathable"]),
    ("Trucker Hat", ["mesh","casual"]),
]

CAT_TABLE = {"shoes": SHOE_TYPES, "bags": BAG_TYPES, "jackets": JACKET_TYPES, "caps": CAP_TYPES}

ADJ_MATERIAL = [
    "breathable", "water-resistant", "lightweight", "durable", "soft-touch",
    "recycled-blend", "quick-dry", "premium knit", "all-season"
]
ADJ_FEATURE = [
    "all-day comfort", "secure fit", "everyday practicality", "versatile style",
    "travel-ready design", "easy care", "modern silhouette"
]

def _pick_type_and_tags(cat: str, filename: str) -> Tuple[str, List[str]]:
    n = filename.lower()
    table = CAT_TABLE[cat]
    for title, tags in table:
        if any(k in n for k in tags):
            return title, tags
    # deterministic pick based on file hash (so builds are stable)
    h = int(hashlib.md5(filename.encode("utf-8")).hexdigest(), 16)
    return table[h % len(table)]

def _varied_description(cat: str, color: str, title_variant: str, tags: List[str], seed: str) -> str:
    h = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16)
    mat = ADJ_MATERIAL[h % len(ADJ_MATERIAL)]
    feat = ADJ_FEATURE[(h // 7) % len(ADJ_FEATURE)]
    purpose = ", ".join(tags[:2]) if tags else "everyday"
    cat_singular = cat[:-1] if cat.endswith("s") else cat
    return (
        f"{color.capitalize()} {title_variant} crafted with {mat} materials for {purpose}. "
        f"Offers {feat}. A reliable {cat_singular} you can use daily."
    )

# ---- Price heuristic (kept) ----
BASE_PRICE = {"bags": 49.0, "shoes": 69.0, "jackets": 99.0, "caps": 19.0}

def ensure_catalog(data_dir: Path, cache_dir: Path, regenerate: bool=False) -> Path:
    """
    Scan data/<category>/* and build a rich catalog:
      id, title, category, color, price, description, tags, image_path
    Titles & descriptions vary per item; color detection is robust.
    """
    out_path = cache_dir / "catalog.json"
    if out_path.exists() and not regenerate:
        return out_path

    catalog: List[Dict[str,Any]] = []
    id_counter = 1

    for cat_folder in data_dir.iterdir():
        if not cat_folder.is_dir(): continue
        cat_norm = CAT_MAP.get(cat_folder.name.lower())
        if not cat_norm: continue

        for p in cat_folder.glob("*"):
            if not p.is_file(): continue
            if p.suffix.lower() not in (".jpg",".jpeg",".png",".webp"): continue

            fname = p.stem
            # 1) Prefer filename color, else detect robustly
            color = _color_from_filename(fname) or _dominant_color_name(p)

            # 2) Choose a sensible type & tags (deterministic)
            title_variant, tags = _pick_type_and_tags(cat_norm, fname)

            # 3) Build a varied, realistic title/description
            title = f"{color.capitalize()} {title_variant}"
            desc = _varied_description(cat_norm, color, title_variant, tags, seed=fname)

            # 4) Price with tiny jitter (deterministic)
            base = BASE_PRICE[cat_norm]
            jitter = (abs(hash(fname)) % 1500)/100.0  # 0..15
            price = round(base + jitter, 2)

            catalog.append({
                "id": f"item-{id_counter}",
                "title": title,
                "category": cat_norm,
                "color": color,
                "price": price,
                "description": desc,
                "tags": tags,
                "image_path": str(p.relative_to(data_dir)),
            })
            id_counter += 1

    out_path.write_text(json.dumps(catalog, indent=2))
    return out_path
