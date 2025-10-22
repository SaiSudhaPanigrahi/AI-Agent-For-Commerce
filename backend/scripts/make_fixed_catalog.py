#!/usr/bin/env python3
# Generates a fixed 54-item catalog (6 categories × 3 types × 3 colors)
# Images are embedded SVG data-URIs (no network, never blank).

import json, os, random, hashlib
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(__file__))  # .../backend
CATALOG_PATH = os.path.join(ROOT, "data", "catalog.json")

BRANDS = ["Aero", "Stratus", "CoreWear", "Northline", "UrbanWalk", "FlowFit"]

PALETTE = {
    "black":  "#0b0b0f",
    "white":  "#f7f7fb",
    "blue":   "#1f6feb",
}

# category -> 3 types
TYPES: Dict[str, List[str]] = {
    "shoes":   ["running", "sneakers", "boots"],
    "shirts":  ["t-shirt", "polo", "dress-shirt"],
    "pants":   ["jeans", "chinos", "joggers"],
    "jackets": ["denim-jacket", "bomber", "rain-jacket"],
    "caps":    ["baseball-cap", "beanie", "bucket-hat"],
    "dresses": ["summer", "cocktail", "maxi"],
}

COLORS = ["black", "white", "blue"]

def svg_data_uri(label: str, sublabel: str, hex_color: str) -> str:
    # neon-ish dark card with big text
    text_color = "#e5e7eb" if hex_color != "#f7f7fb" else "#111827"
    border = "#10b981"
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='800' height='800'>
  <defs>
    <linearGradient id='g' x1='0' x2='1'>
      <stop offset='0%' stop-color='{hex_color}'/>
      <stop offset='100%' stop-color='{hex_color}' stop-opacity='0.9'/>
    </linearGradient>
  </defs>
  <rect x='0' y='0' width='800' height='800' fill='url(#g)' />
  <rect x='12' y='12' width='776' height='776' fill='none' stroke='{border}' stroke-width='6' rx='28'/>
  <g font-family='-apple-system, Inter, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif' text-anchor='middle'>
    <text x='400' y='380' font-size='64' fill='{text_color}' font-weight='700'>{label}</text>
    <text x='400' y='460' font-size='36' fill='{text_color}' opacity='0.85'>{sublabel}</text>
  </g>
</svg>"""
    import urllib.parse
    encoded = urllib.parse.quote(svg)
    return f"data:image/svg+xml;charset=utf-8,{encoded}"

def price_for(category: str, type_: str) -> float:
    base = {
        "shoes": 69, "shirts": 29, "pants": 49,
        "jackets": 89, "caps": 19, "dresses": 59
    }[category]
    # tiny deterministic bump per type
    bump = (int(hashlib.md5(type_.encode()).hexdigest(), 16) % 700) / 100.0
    return round(base + (bump % 10), 2)

def make_item(cat: str, type_: str, color: str, idx: int) -> Dict:
    brand = BRANDS[idx % len(BRANDS)]
    title = f"{brand} {type_.replace('-', ' ').title()} — {color.title()}"
    desc  = f"{color.title()} {type_.replace('-', ' ')} in {cat}; breathable, durable, everyday style."
    tags  = [cat, type_, color, "everyday", "neon-card"]
    image = svg_data_uri(type_.replace('-', ' ').title(), color.title(), PALETTE[color])

    return {
        "id": f"{cat[:2]}-{type_[:3]}-{color[:2]}-{idx}",
        "title": title,
        "brand": brand,
        "category": cat,
        "price": price_for(cat, type_),
        "description": desc,
        "tags": tags,
        "image": image   # NOTE: we use `image` (schemas allows alias to image_url if you wired that)
    }

def main():
    items: List[Dict] = []
    idx = 0
    for cat, type_list in TYPES.items():
        for type_ in type_list:
            for color in COLORS:
                items.append(make_item(cat, type_, color, idx))
                idx += 1

    os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
    with open(CATALOG_PATH, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Wrote {len(items)} products to {CATALOG_PATH}")

if __name__ == "__main__":
    main()
