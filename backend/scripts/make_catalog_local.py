#!/usr/bin/env python3
import json
import re
from pathlib import Path
from statistics import mean
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"
CATALOG = DATA / "catalog.json"

CATEGORY_MAP = {
    "bag": ("bags", "CarryCo"),
    "shoe": ("shoes", "Nimbus"),
    "jacket": ("jackets", "TrailCore"),
    "cap": ("caps", "Brightline"),
}

# Simple color names by nearest of these anchors
NAMED_COLORS = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "red": (200, 40, 40),
    "orange": (230, 120, 30),
    "yellow": (230, 220, 60),
    "green": (60, 170, 80),
    "blue": (70, 100, 210),
    "purple": (140, 80, 180),
    "brown": (140, 100, 60),
    "pink": (230, 140, 180),
}

def closest_color(rgb):
    r, g, b = rgb
    best = None
    best_d = 10**9
    for name, (R, G, B) in NAMED_COLORS.items():
        d = (R - r) ** 2 + (G - g) ** 2 + (B - b) ** 2
        if d < best_d:
            best_d, best = d, name
    return best

def dominant_color(path: Path):
    img = Image.open(path).convert("RGB")
    img = img.resize((64, 64))  # speed / smoothing
    pixels = list(img.getdata())
    # ignore very bright/very dark extremes to avoid white background bias
    filtered = [(r, g, b) for (r, g, b) in pixels if 10 < r < 245 or 10 < g < 245 or 10 < b < 245]
    if not filtered:
        filtered = pixels
    r = int(mean([p[0] for p in filtered]))
    g = int(mean([p[1] for p in filtered]))
    b = int(mean([p[2] for p in filtered]))
    return closest_color((r, g, b))

def price_for(category: str, index: int) -> float:
    # stagger prices per category
    base = {"bags": 49, "shoes": 79, "jackets": 99, "caps": 24}.get(category, 39)
    return float(base + (index % 5) * 5)  # 5-dollar steps

def title_for(category: str, color: str):
    if category == "bags":
        return f"{color.title()} City Tote"
    if category == "shoes":
        return f"{color.title()} Street Sneaker"
    if category == "jackets":
        return f"{color.title()} Windbreaker Jacket"
    if category == "caps":
        return f"{color.title()} Classic Cap"
    return f"{color.title()} Item"

def description_for(category: str, color: str):
    blurb = {
        "bags": f"{color.title()} everyday carry with interior pockets and durable canvas.",
        "shoes": f"{color.title()} sneaker with cushioned midsole and grippy rubber outsole.",
        "jackets": f"Lightweight {color} jacket with wind resistant shell and soft lining.",
        "caps": f"Unstructured {color} cap with curved brim and comfy fit.",
    }
    return blurb[category]

def scan():
    items = []
    rx = re.compile(r"^(bag|shoe|jacket|cap)(\d+)\.(jpg|jpeg|png)$", re.I)
    for p in sorted(IMAGES.glob("*")):
        m = rx.match(p.name)
        if not m:
            continue
        stem, idx = m.group(1).lower(), int(m.group(2))
        category, brand = CATEGORY_MAP[stem]
        color = dominant_color(p)
        price = price_for(category, idx)
        title = title_for(category, color)
        desc = description_for(category, color)
        item = {
            "id": f"{category}-{idx}",
            "title": title,
            "brand": brand,
            "category": category,
            "color": color,
            "price": round(price, 2),
            "description": desc,
            "image": f"/images/{p.name}",
        }
        items.append(item)
    return items

def main():
    items = scan()
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    print(f"wrote {len(items)} items → {CATALOG}")

if __name__ == "__main__":
    main()
