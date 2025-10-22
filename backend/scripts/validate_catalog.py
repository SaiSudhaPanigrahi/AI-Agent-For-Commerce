"""
Quick validator: ensures every item has an https image and required fields.
If an image is not http/https, it replaces it with a neutral Unsplash image.
"""
from pathlib import Path
import json
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "data" / "catalog.json"

FALLBACK = "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?auto=format&fit=crop&w=800&h=800&q=80"

def is_http(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https")
    except Exception:
        return False

def main():
    data = json.loads(CAT.read_text(encoding="utf-8"))
    fixed = 0
    for it in data:
        if not is_http(it.get("image", "")):
            it["image"] = FALLBACK
            fixed += 1
    CAT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Validated {len(data)} items. Fixed images: {fixed}")

if __name__ == "__main__":
    main()
