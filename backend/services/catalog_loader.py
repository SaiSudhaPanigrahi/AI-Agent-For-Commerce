from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import re, json, colorsys
from PIL import Image
import numpy as np

CAT_MAP = {
    "bag":"bags","bags":"bags",
    "shoe":"shoes","shoes":"shoes",
    "jacket":"jackets","jackets":"jackets",
    "cap":"caps","caps":"caps",
}
COLOR_NAMES = ["red","blue","green","black","white","yellow","brown","gray","purple","orange","assorted"]

def _dominant_color_name(img_path: Path) -> str:
    try:
        im = Image.open(img_path).convert("RGB").resize((64,64))
        arr = np.asarray(im).astype(np.float32)/255.0
        hsv_sum = np.zeros(3, dtype=np.float32)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                hsv_sum += np.array(colorsys.rgb_to_hsv(arr[y,x,0],arr[y,x,1],arr[y,x,2]), dtype=np.float32)
        mean = hsv_sum/(arr.shape[0]*arr.shape[1])
        h = mean[0]*360
        s = mean[1]; v = mean[2]
        # quick snap to coarse bins
        if s < 0.1 and v > 0.9: return "white"
        if s < 0.12 and v < 0.25: return "black"
        if s < 0.15: return "gray"
        if   345<=h or h<15: return "red"
        elif 15<=h<45: return "orange"
        elif 45<=h<75: return "yellow"
        elif 75<=h<165: return "green"
        elif 165<=h<255: return "blue"
        elif 255<=h<315: return "purple"
        elif 315<=h<345: return "red"
        return "assorted"
    except Exception:
        return "assorted"

def _name_color_from_filename(name: str) -> str | None:
    n = name.lower()
    for c in COLOR_NAMES:
        if c != "assorted" and re.search(rf"\b{c}\b", n):
            return c
    return None

def ensure_catalog(data_dir: Path, cache_dir: Path, regenerate: bool=False) -> Path:
    """
    Scans data_dir/<category>/* for images and builds a catalog.json with:
    id, title, category, color, price, description, image_path
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
            color = _name_color_from_filename(fname) or _dominant_color_name(p)

            # Simple price heuristic by category (override later if needed)
            base_price = {
                "bags": 49.0, "shoes": 69.0, "jackets": 99.0, "caps": 19.0
            }[cat_norm]
            # add a tiny jitter by filename hash
            jitter = (abs(hash(fname)) % 1500)/100.0  # 0..15
            price = round(base_price + jitter, 2)

            item = {
                "id": f"item-{id_counter}",
                "title": f"{color.capitalize()} {cat_norm[:-1].capitalize()}",
                "category": cat_norm,
                "color": color,
                "price": price,
                "description": f"A {color} {cat_norm[:-1]} from our curated catalog.",
                "image_path": str(p.relative_to(data_dir)),
            }
            catalog.append(item)
            id_counter += 1

    out_path.write_text(json.dumps(catalog, indent=2))
    return out_path
