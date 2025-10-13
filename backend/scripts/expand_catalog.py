import json, os, random, hashlib, itertools, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def unsplash(url):  # normalized size, crop & quality for speed + consistency
    join = "&" if "?" in url else "?"
    return f"{url}{join}auto=format&fit=crop&w=1200&q=80"

UNSPLASH_BY_CAT = {
    "t-shirts": [
        unsplash("https://images.unsplash.com/photo-1516826957135-700dedea698c"),
        unsplash("https://images.unsplash.com/photo-1520975916090-3105956dac38"),
        unsplash("https://images.unsplash.com/photo-1520975682034-4e3aa7b2c0f8"),
    ],
    "shoes": [
        unsplash("https://images.unsplash.com/photo-1608231387042-66d1773070a5"),
        unsplash("https://images.unsplash.com/photo-1542291026-7eec264c27ff"),
        unsplash("https://images.unsplash.com/photo-1520975916090-3105956dac38"),
    ],
    "shorts": [
        unsplash("https://images.unsplash.com/photo-1603252109334-43d2011b4dde"),
        unsplash("https://images.unsplash.com/photo-1523381210434-271e8be1f52b"),
    ],
    "hoodies": [
        unsplash("https://images.unsplash.com/photo-1521572163474-6864f9cf17ab"),
        unsplash("https://images.unsplash.com/photo-1490481651871-ab68de25d43d"),
    ],
    "tanks": [
        unsplash("https://images.unsplash.com/photo-1598978224661-5cf9ddb4ba44"),
        unsplash("https://images.unsplash.com/photo-1532074205216-d0e1f4b87368"),
    ],
    "leggings": [
        unsplash("https://images.unsplash.com/photo-1549570652-97324981a6fd"),
        unsplash("https://images.unsplash.com/photo-1534367610401-9f51e1b0a7f1"),
    ],
    "bags": [
        unsplash("https://images.unsplash.com/photo-1548013146-72479768bada"),
        unsplash("https://images.unsplash.com/photo-1500043357865-c6b8827edf24"),
    ],
    "bottles": [
        unsplash("https://images.unsplash.com/photo-1598986646512-3f2b36ebd9d8"),
        unsplash("https://images.unsplash.com/photo-1532634896-26909d0d4b6a"),
    ],
}

BRANDS = {
    "t-shirts": ["Aero", "Stratus", "CoreWear"],
    "shoes": ["UrbanWalk", "WildStep"],
    "shorts": ["CoreWear"],
    "hoodies": ["Northline"],
    "tanks": ["Aero"],
    "leggings": ["FlowFit"],
    "bags": ["Transit"],
    "bottles": ["HydroPeak"],
}

BASE_PRODUCTS = [
    ("AeroDry Run Tee", "t-shirts", "Lightweight moisture-wicking tee ideal for running and gym.", 24.99),
    ("Breeze Performance Tee", "t-shirts", "Ultra-breathable mesh tee for high-intensity workouts.", 27.99),
    ("CityStride Sneakers", "shoes", "Everyday lifestyle sneakers with cushioned midsole.", 69.00),
    ("TrailMax Sneakers", "shoes", "Rugged trail-running shoes with aggressive grip.", 79.00),
    ("FlexFit Training Shorts", "shorts", "4-way stretch training shorts with zip pocket.", 29.50),
    ("HeatGuard Hoodie", "hoodies", "Midweight hoodie with brushed fleece interior.", 49.00),
    ("SwiftDry Tank", "tanks", "Sweat-wicking training tank with racerback fit.", 22.00),
    ("Commuter Backpack 20L", "bags", "Water-resistant backpack with padded laptop sleeve.", 59.00),
    ("Studio Training Leggings", "leggings", "High-rise leggings with squat-proof fabric.", 39.00),
    ("ThermaBottle 750ml", "bottles", "Double-wall insulated bottle keeps drinks cold 24h.", 19.99),
]

COLORS = ["Black", "White", "Navy", "Gray", "Olive", "Red"]
SIZES_BY_CAT = {
    "t-shirts": ["S", "M", "L", "XL"],
    "hoodies": ["S", "M", "L", "XL"],
    "tanks": ["S", "M", "L", "XL"],
    "leggings": ["XS", "S", "M", "L"],
    "shorts": ["S", "M", "L", "XL"],
    "shoes": ["7", "8", "9", "10", "11"],
    "bags": ["20L"],
    "bottles": ["750ml"],
}

def pid(*parts):
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]

def pick_img(cat, seed_idx):
    pool = UNSPLASH_BY_CAT.get(cat, [])
    if not pool:
        pool = [UNSPLASH_BY_CAT["t-shirts"][0]]
    return pool[seed_idx % len(pool)]

def build():
    random.seed(42)
    items = []
    for title, cat, desc, base_price in BASE_PRODUCTS:
        brand = BRANDS.get(cat, ["Mercury"])[0]
        sizes = SIZES_BY_CAT[cat]
        variants = list(itertools.product(COLORS[:3], sizes[:3]))  # 9 variants each base -> ~90+ products total
        for i, (color, size) in enumerate(variants):
            price_jitter = (random.random() - 0.5) * (0.08 * base_price)  # ±8%
            price = round(base_price + price_jitter, 2)
            full_title = f"{title} ({color}, {size})" if cat not in ["bags", "bottles"] else f"{title} ({color})"
            image_url = pick_img(cat, i)
            item = {
                "id": pid(title, cat, color, size),
                "title": full_title,
                "category": cat,
                "description": desc,
                "price": price,
                "image_url": image_url,
                "brand": brand,
                "tags": [color.lower(), size.lower()] if size else [color.lower()],
            }
            items.append(item)
    # small dedupe by title if any collisions happen
    seen = set()
    dedup = []
    for it in items:
        if it["id"] not in seen:
            dedup.append(it); seen.add(it["id"])
    return dedup

if __name__ == "__main__":
    out = build()
    (DATA / "catalog.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} products to {DATA/'catalog.json'}")
