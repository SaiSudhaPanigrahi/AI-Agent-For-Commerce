# backend/services/enricher.py
from __future__ import annotations
import os, json, re, argparse, hashlib, random
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None

BASIC_COLORS = [
    "black","white","red","blue","green","yellow","orange","purple",
    "pink","brown","grey","gray","beige","navy","teal"
]
COLOR_SYNONYMS = {
    "crimson":"red","maroon":"red","burgundy":"red",
    "navy":"blue","sky":"blue","cobalt":"blue","teal":"blue",
    "charcoal":"grey","gray":"grey","slate":"grey",
    "ivory":"white","off-white":"white","cream":"white",
    "tan":"beige","camel":"beige","khaki":"beige",
    "magenta":"pink","rose":"pink","fuchsia":"pink",
    "forest":"green","olive":"green",
}

CAT_SET = {"bags","caps","jackets","shoes"}

# Category-specific vocab pools (traits, use-cases, constructions)
POOLS = {
    "bags": {
        "styles": ["tote","crossbody","sling","backpack","satchel","mini tote","duffel","messenger"],
        "features": ["zip top","inner organizer","padded strap","quick-access pocket",
                     "water-resistant shell","durable canvas","premium leather trim","lightweight build"],
        "uses": ["daily carry","commute","travel","gym days","weekend strolls","errands"],
        "audience": ["women","men","unisex"],
    },
    "caps": {
        "styles": ["baseball cap","5-panel","snapback","dad cap","trucker","beanie"],
        "features": ["curved brim","breathable eyelets","adjustable strap","mesh back",
                     "moisture-wicking band","low-profile crown"],
        "uses": ["daily wear","outdoor runs","sunny days","casual fits","weekend errands"],
        "audience": ["women","men","unisex"],
    },
    "jackets": {
        "styles": ["windbreaker","puffer","lightweight shell","coach jacket","hooded shell","fleece-lined jacket"],
        "features": ["water-resistant finish","windproof paneling","zip pockets","packable hood",
                     "breathable mesh lining","elastic cuffs","thermal insulation"],
        "uses": ["sporty commutes","runs in chill weather","travel layers","weekend hikes","everyday wear"],
        "audience": ["women","men","unisex"],
    },
    "shoes": {
        "styles": ["trainer","runner","sneaker","road runner","trail sneaker","court-style"],
        "features": ["cushioned midsole","breathable mesh","supportive heel cup","grippy outsole",
                     "lightweight foam","flex grooves","arch support"],
        "uses": ["running","gym sessions","daily miles","weekend walks","travel days"],
        "audience": ["women","men","unisex"],
    },
}

# Terms we’ll turn into tags if they appear in user-facing text
INTENT_TERMS = [
    "running","trail","hiking","gym","workout","sport","sporty","formal","casual",
    "winter","summer","waterproof","leather","canvas","breathable","lightweight",
    "women","ladies","men","unisex","commute","travel","daily","weekend"
]

# Gemini
PROMPT = """You are improving product copy for an online shop.

Input JSON has: title, category, base color, any current description/tags.
Return a JSON with:
{
  "description": "1–2 punchy sentences highlighting use cases and key features (no brand/specs).",
  "tags": ["5-10 lowercase tags, e.g., 'running','women','lightweight','waterproof'"]
}
Keep it grounded in the input. Do not invent specs or materials not implied by the category.
Return ONLY JSON.
"""

def _hash_seed(*parts: str) -> int:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _norm_color(text: str, fallback: Optional[str]) -> Optional[str]:
    t = (text or "").lower()
    for k, v in COLOR_SYNONYMS.items():
        if re.search(rf"\b{k}\b", t): return v
    for c in BASIC_COLORS:
        if re.search(rf"\b{c}\b", t): return "grey" if c == "gray" else c
    if isinstance(fallback, str) and fallback:
        c = fallback.lower()
        return "grey" if c == "gray" else c
    return None

def _detect_category(cat: Optional[str], title: str, desc: str) -> Optional[str]:
    c = (cat or "").lower()
    if c in CAT_SET: return c
    blob = f"{title} {desc}".lower()
    if re.search(r"\bbag|tote|handbag|backpack|crossbody|sling|duffel|satchel\b", blob): return "bags"
    if re.search(r"\bcap|snapback|beanie|hat\b", blob): return "caps"
    if re.search(r"\bjacket|windbreaker|puffer|shell|parka|blazer|coach\b", blob): return "jackets"
    if re.search(r"\bshoe|sneaker|trainer|runner|boot\b", blob): return "shoes"
    return None

def _choose_unique(cat: str, seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    p = POOLS[cat]
    style = rng.choice(p["styles"])
    feat1, feat2 = rng.sample(p["features"], 2)
    use = rng.choice(p["uses"])
    audience = rng.choice(p["audience"])
    return {"style": style, "feat1": feat1, "feat2": feat2, "use": use, "audience": audience}

def _compose_desc(title: str, cat: str, color: Optional[str], picks: Dict[str,str]) -> str:
    # “Assorted bag — durable, roomy…” becomes richer & unique per item
    col = f"{color} " if color else ""
    noun = cat[:-1]  # bags -> bag
    s1 = f"{title}: a {picks['style']} {col}{noun} built for {picks['use']}."
    s2 = f"Features {picks['feat1']} and {picks['feat2']} with a {picks['audience']}-friendly fit."
    return f"{s1} {s2}"

def _augment_tags(base: List[str], cat: str, color: Optional[str], picks: Dict[str,str], existing_text: str) -> List[str]:
    tags = [t.lower() for t in base if isinstance(t, str)]
    tags += [picks["use"], picks["audience"], picks["style"]]
    if color: tags.append(color)
    tags.append(cat)
    # add terms we spot in existing text
    blob = existing_text.lower()
    for t in INTENT_TERMS:
        if t in blob: tags.append(t)
    # de-dup & cap
    out = []
    for t in tags:
        if t and t not in out: out.append(t)
    return out[:12]

def _call_gemini(model, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        resp = model.generate_content(PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False),
                                      request_options={"timeout": 25})
        txt = (getattr(resp, "text", None) or "").strip()
        if txt.startswith("{") and '"tags"' in txt and '"description"' in txt:
            data = json.loads(txt)
            # sanitize
            descr = str(data.get("description","")).strip()
            tags = [str(x).lower().strip() for x in data.get("tags", []) if str(x).strip()]
            return {"description": descr, "tags": tags[:12]}
    except Exception:
        return None
    return None

def enrich_catalog(catalog_path: Path, out_path: Optional[Path] = None) -> Path:
    data = json.loads(catalog_path.read_text())
    out: List[Dict[str, Any]] = []

    # Optional Gemini
    model = None
    if genai is not None and os.getenv("GOOGLE_API_KEY"):
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            for name in ["models/gemini-1.5-flash-8b","models/gemini-1.5-flash-002","models/gemini-1.5-flash"]:
                try:
                    m = genai.GenerativeModel(name)
                    _ = m.generate_content("ok", request_options={"timeout": 5})
                    model = m; break
                except Exception:
                    continue
        except Exception:
            model = None

    for it in data:
        title = it.get("title","Item")
        cat0 = it.get("category")
        desc0 = it.get("description","")
        color0 = it.get("color")
        img = str(it.get("image_path",""))

        # derive color from filename if needed
        guess_from_name = _norm_color(Path(img).stem.replace("_"," ").replace("-"," "), None)
        color = _norm_color(f"{title} {desc0}", color0) or guess_from_name

        # normalize category if missing/incorrect
        cat = _detect_category(cat0, title, desc0) or (cat0 if cat0 in CAT_SET else None)
        if not cat:
            # fallback: infer from folder name if present
            folder = Path(img).parent.name.lower()
            if folder in CAT_SET: cat = folder
        if not cat:
            # if still None, default to bags (most neutral)
            cat = "bags"

        # deterministic picks ensure uniqueness (but stable across runs)
        seed = _hash_seed(str(it.get("id","")), title, img, cat, str(color or ""))
        picks = _choose_unique(cat, seed)

        # ---- Try Gemini first ----
        gem_payload = {
            "title": title, "category": cat, "color": color,
            "current_description": desc0, "existing_tags": it.get("tags", []),
        }
        enriched = _call_gemini(model, gem_payload) if model else None

        # If Gemini gave generic/short text, or no tags, augment with deterministic flavor
        if not enriched or len(enriched.get("description","").split()) < 8 or len(enriched.get("tags",[])) < 4:
            # Compose our unique description
            local_desc = _compose_desc(title, cat, color, picks)
            # Merge: prefer Gemini if present, but strengthen it
            desc = (enriched.get("description") if enriched else "") or local_desc
            tags_base = (enriched.get("tags") if enriched else []) + (it.get("tags", []) or [])
            tags = _augment_tags(tags_base, cat, color, picks, desc + " " + desc0)
        else:
            # Enriched is good; still add uniqueness via tags and ensure color/cat present
            desc = enriched["description"]
            tags = _augment_tags(enriched["tags"] + (it.get("tags",[]) or []), cat, color, picks, desc + " " + desc0)

        # Update item
        it["category"] = cat
        if color: it["color"] = color
        it["description"] = desc
        it["tags"] = tags
        out.append(it)

    dst = out_path or catalog_path
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="backend/data/catalog.json")
    ap.add_argument("--out", dest="out", default="")
    args = ap.parse_args()
    src = Path(args.inp)
    out = Path(args.out) if args.out else src
    dst = enrich_catalog(src, out)
    print(f"Enriched catalog saved → {dst}")

if __name__ == "__main__":
    main()
