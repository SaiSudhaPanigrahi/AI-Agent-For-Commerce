# -*- coding: utf-8 -*-
"""
Build a 54-item catalog with stable, real Unsplash images:
- 6 categories × 3 types × 3 colors (black/white/blue)
Images are fixed by ID (not random) and served via https.
"""
import json
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "catalog.json"

# Stable Unsplash photo IDs by (category, type)
# Note: these are generic but visually relevant per type.
UNSPLASH = {
    ("shoes", "sneakers"): "photo-1519741497674-611481863552",
    ("shoes", "running"):  "photo-1520975916090-3105956dac38",
    ("shoes", "boots"):    "photo-1542291026-7eec264c27ff",

    ("shirts", "t-shirt"): "photo-1512436991641-6745cdb1723f",
    ("shirts", "polo"):    "photo-1503342217505-b0a15cf70489",
    ("shirts", "dress"):   "photo-1520975916090-3105956dac38",

    ("pants", "jeans"):    "photo-1516826957135-700dedea698c",
    ("pants", "chinos"):   "photo-1541099649105-f69ad21f3246",
    ("pants", "joggers"):  "photo-1460353581641-37baddab0fa2",

    ("jackets", "bomber"): "photo-1520975916090-3105956dac38",
    ("jackets", "puffer"): "photo-1512436991641-6745cdb1723f",
    ("jackets", "denim"):  "photo-1520975916090-3105956dac38",

    ("caps", "baseball"):  "photo-1512436991641-6745cdb1723f",
    ("caps", "trucker"):   "photo-1519741497674-611481863552",
    ("caps", "beanie"):    "photo-1460353581641-37baddab0fa2",

    ("dresses", "casual"): "photo-1539008835657-9e8e8d0f8c1a",
    ("dresses", "midi"):   "photo-1520975916090-3105956dac38",
    ("dresses", "formal"): "photo-1520975916090-3105956dac38",
}

CATEGORIES = {
    "shoes":   ["sneakers", "running", "boots"],
    "shirts":  ["t-shirt", "polo", "dress"],
    "pants":   ["jeans", "chinos", "joggers"],
    "jackets": ["bomber", "puffer", "denim"],
    "caps":    ["baseball", "trucker", "beanie"],
    "dresses": ["casual", "midi", "formal"],
}

COLORS = ["black", "white", "blue"]

def uurl(photo_id: str, w=800, h=800, q=80) -> str:
    # Fixed crop/size with Unsplash processing params
    return f"https://images.unsplash.com/{photo_id}?auto=format&fit=crop&w={w}&h={h}&q={q}"

def price_for(cat: str, typ: str) -> float:
    base = {
        "shoes": 49.0, "shirts": 24.0, "pants": 39.0,
        "jackets": 69.0, "caps": 16.0, "dresses": 59.0,
    }[cat]
    bump = {"running": 10, "dress": 8, "boots": 12, "puffer": 12, "formal": 15}.get(typ, 0)
    return round(base + bump + (hash(typ) % 5), 2)

def make_item(cat: str, typ: str, color: str, idx: int) -> dict:
    pid = UNSPLASH.get((cat, typ))
    img = uurl(pid) if pid else uurl("photo-1512436991641-6745cdb1723f")
    brand = {
        "shoes": "StrideX",
        "shirts": "UrbanWalk",
        "pants": "Aero",
        "jackets": "Northline",
        "caps": "CoreWear",
        "dresses": "FlowFit",
    }[cat]
    title = f"{brand} {typ.title()} — {color.title()}"
    desc = f"{color} {typ} in {cat}; breathable, durable, everyday style."
    return {
        "id": f"{cat[:2]}-{typ[:2]}-{color[:2]}-{idx}",
        "title": title,
        "brand": brand,
        "category": cat,
        "color": color,
        "price": price_for(cat, typ),
        "description": desc,
        "image": img,
        "tags": [cat, typ, color, "everyday", "breathable", "durable"],
    }

def main():
    items = []
    idx = 0
    for cat, types in CATEGORIES.items():
        for typ, col in product(types, COLORS):
            idx += 1
            items.append(make_item(cat, typ, col, idx))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} products to {OUT}")

if __name__ == "__main__":
    main()
