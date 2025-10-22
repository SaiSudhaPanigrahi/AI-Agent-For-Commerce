
import re
from typing import List, Dict, Any
from .rag import RAGIndex

_COLOR_WORDS = [
    "red","blue","green","black","white","pink","brown","beige","teal","navy","grey","gray","yellow","purple","orange"
]
_CAT_WORDS = {
    "bag":"bags","bags":"bags",
    "shoe":"shoes","shoes":"shoes",
    "jacket":"jackets","jackets":"jackets",
    "cap":"caps","caps":"caps",
    "top":"tops","tops":"tops",
    "dress":"dresses","dresses":"dresses",
    "pant":"pants","pants":"pants",
}

_PRICE_UNDER = re.compile(r"under\s*\$?(\d+)|<=\s*\$?(\d+)", re.I)
_PRICE_OVER = re.compile(r"over\s*\$?(\d+)|>=\s*\$?(\d+)", re.I)

class HybridRecommender:
    def __init__(self, products: List[Dict[str, Any]]):
        self.products = products
        self.rag = RAGIndex(products)

    def _apply_filters(self, items: List[Dict[str,Any]], q: str) -> List[Dict[str,Any]]:
        ql = q.lower()
        # color
        colors = [c for c in _COLOR_WORDS if c in ql]
        if colors:
            items = [p for p in items if p.get("color") and any(c in p["color"].lower() for c in colors)]
        # category
        cats = [v for k,v in _CAT_WORDS.items() if k in ql]
        if cats:
            items = [p for p in items if p.get("category") in cats]
        # price
        price_max = None
        m = _PRICE_UNDER.search(ql)
        if m:
            price_max = float(m.group(1) or m.group(2))
        price_min = None
        m = _PRICE_OVER.search(ql)
        if m:
            price_min = float(m.group(1) or m.group(2))
        if price_max is not None:
            items = [p for p in items if float(p.get("price", 0)) <= price_max]
        if price_min is not None:
            items = [p for p in items if float(p.get("price", 0)) >= price_min]
        return items

    def recommend(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        # initial RAG retrieval
        cand = self.rag.search_products(query, k=k*2)
        # filters
        cand = self._apply_filters(cand, query)
        return cand[:k]
