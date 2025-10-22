from pathlib import Path
import json, random

CATEGORIES = ["bags","caps","jackets","shoes"]
COLORS = ["red","blue","green","black","white","yellow","brown","gray","purple","orange"]

def _infer_color(name: str) -> str:
    lname = name.lower()
    for c in COLORS:
        if c in lname:
            return c
    return "assorted"

def _make_desc(cat: str, color: str) -> str:
    extras = {
        "bags": "durable, roomy, perfect for daily use",
        "caps": "breathable, adjustable, ideal for outdoors",
        "jackets": "warm, lightweight, great for layering",
        "shoes": "comfortable, supportive, everyday wear",
    }
    return f"{color.capitalize()} {cat[:-1]} — {extras.get(cat, 'quality material')}."

def _price_for(cat: str) -> float:
    ranges = {
        "bags": (25, 120),
        "caps": (8, 35),
        "jackets": (40, 180),
        "shoes": (30, 150),
    }
    lo, hi = ranges.get(cat, (10,100))
    return round(random.uniform(lo, hi), 2)

def scan_images(data_dir: Path):
    items = []
    pid = 1
    for cat in CATEGORIES:
        folder = data_dir / cat
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.suffix.lower() not in [".jpg",".jpeg",".png",".webp"]:
                continue
            color = _infer_color(p.stem)
            title = f"{color.capitalize()} {cat[:-1]}"
            item = {
                "id": pid,
                "title": title,
                "category": cat,
                "color": color,
                "price": _price_for(cat),
                "description": _make_desc(cat, color),
                "image_path": str(p.relative_to(data_dir)),
                "tags": [color, cat, title.lower()]
            }
            items.append(item)
            pid += 1
    return items

def ensure_catalog(data_dir: Path, cache_dir: Path, regenerate: bool=False) -> Path:
    cache_dir.mkdir(exist_ok=True, parents=True)
    catalog_path = cache_dir / "catalog.json"
    if catalog_path.exists() and not regenerate:
        return catalog_path
    items = scan_images(data_dir)
    if not items:
        items = []
    with catalog_path.open("w") as f:
        json.dump(items, f, indent=2)
    return catalog_path
