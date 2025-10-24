from __future__ import annotations
from pathlib import Path
import json, re
from typing import Dict, Any, Optional

ALLOWED_CATS = {"bags","caps","jackets","shoes"}
EXTS = (".jpg",".jpeg",".png",".webp")

def singular(cat: str) -> str:
    return cat[:-1] if cat.endswith("s") else cat

def _candidates(it: Dict[str,Any]) -> list[str]:
    out = []
    for k in ("image_path","image"):
        v = it.get(k)
        if v: out += [Path(v).name, Path(v).stem]
    m = re.match(r"^(bags|caps|jackets|shoes)-(\d+)$", str(it.get("id","")).lower())
    if m:
        out.append(f"{singular(m.group(1))}{m.group(2)}")
    # unique
    seen, uniq = set(), []
    for x in out:
        if x and x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

def _find(data_dir: Path, stem_or_name: str, category_hint: Optional[str]) -> Optional[Path]:
    p = Path(stem_or_name)
    names = [stem_or_name] if p.suffix.lower() in EXTS else [stem_or_name + e for e in EXTS]
    if category_hint in ALLOWED_CATS:
        for nm in names:
            q = data_dir / category_hint / nm
            if q.exists(): return q
    for cat in ALLOWED_CATS:
        for nm in names:
            q = data_dir / cat / nm
            if q.exists(): return q
    for nm in names:
        q = data_dir / nm
        if q.exists(): return q
    low = stem_or_name.lower()
    for folder in [*(data_dir / c for c in ALLOWED_CATS), data_dir]:
        for f in folder.glob("*"):
            if f.is_file() and f.suffix.lower() in EXTS:
                if f.name.lower() == low or f.stem.lower() == low:
                    return f
    return None

def repair_paths(catalog_path: Path, data_dir: Path) -> int:
    items = json.loads(catalog_path.read_text())
    changed = 0
    for it in items:
        cat = str(it.get("category","")).lower()
        found = None
        for cand in _candidates(it):
            q = _find(data_dir, cand, cat)
            if q:
                found = q; break
        if found:
            rel = found.relative_to(data_dir).as_posix()
            if it.get("image_path") != rel:
                it["image_path"] = rel
                changed += 1
            it.pop("image", None)
    catalog_path.write_text(json.dumps(items, indent=2))
    return changed
