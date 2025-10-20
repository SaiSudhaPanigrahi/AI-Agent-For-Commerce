import re, json
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel
from services.chat import has_openai, openai_chat, local_smalltalk
from services.recommender import HybridRecommender
from services.image_search import ImageSearch
from services.rag import RAG
from services.agent_llm import has_ollama, plan_with_llm, answer_with_llm

class AgentReply(BaseModel):
    mode: str
    message: str
    items: Optional[List[Dict[str, Any]]] = None

class Tools:
    def __init__(self, products: List[Dict[str, Any]], recommender: HybridRecommender, image: ImageSearch, rag: RAG):
        self.products = products
        self.rec = recommender
        self.img = image
        self.rag = rag

    def search_text(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        return self.rec.search(query, top_k=top_k)

    def search_image(self, image_url: str, top_k: int = 8) -> List[Dict[str, Any]]:
        return self.img.search_by_image_url(image_url=image_url, top_k=top_k)

    def qa_rag(self, question: str, hint_ids: Optional[Set[str]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        chunks = self.rag.retrieve(question, k=12, hint_ids=hint_ids)
        ctx = ""
        for i, (c, m, s) in enumerate(chunks, 1):
            t = m.get("title","")
            b = m.get("brand","")
            cat = m.get("category","")
            ctx += f"[{i}] {t} | {b} | {cat}\n{c}\n\n"
        if has_ollama():
            msg = answer_with_llm(question, ctx)
        elif has_openai():
            sys = "Answer using only the provided context. Be concise. Use bullet points. Do not invent items not in context."
            user = f"Question:\n{question}\n\nContext:\n{ctx}"
            msg = openai_chat(user)
        else:
            msg = "Here is what I found:\n" + "\n".join([f"- {m.get('title','')}" for _,m,_ in chunks[:5]])
        top_pids = self.rag.top_products_from_chunks(chunks, top_n=6)
        idset = set(top_pids)
        items = []
        for p in self.products:
            pid = str(p.get("id") or p.get("sku") or p.get("title"))
            if pid in idset:
                items.append(p)
        return msg, items

    def smalltalk(self, msg: str) -> str:
        return local_smalltalk(msg)

class Agent:
    def __init__(self, products: List[Dict[str, Any]], tools: Tools):
        self.products = products
        self.tools = tools

    def _extract_image_url(self, msg: str) -> Optional[str]:
        m = re.search(r'(https?://[^\s]+?\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s]*)?)', msg, re.I)
        if m: return m.group(1)
        if msg.lower().startswith("image url:"):
            u = msg.split(":",1)[1].strip()
            if u.startswith("http"): return u
        return None

    def _has_price_intent(self, msg: str) -> bool:
        return bool(re.search(r'(under|below|less than|<=|\$)\s*\$?\s*\d+', msg.lower()))

    def _has_product_intent(self, msg: str) -> bool:
        k = ["tshirt","t-shirt","tee","shirt","sneaker","shoe","short","hoodie","tank","legging","bottle","bag","backpack","sports","running","gym","workout","trail","compare","which","difference","explain","why"]
        m = msg.lower()
        return any(s in m for s in k)

    def _is_qa_intent(self, msg: str) -> bool:
        m = msg.lower()
        return any(w in m for w in ["compare","difference","which is better","why","explain","vs","versus","good for","sizing","care","wash","returns","policy"])

    def _rule_plan(self, msg: str) -> Tuple[str, Dict[str, Any]]:
        u = self._extract_image_url(msg)
        if u:
            return "image-search", {"image_url": u}
        if self._is_qa_intent(msg):
            return "qa-rag", {"question": msg}
        if self._has_product_intent(msg) or self._has_price_intent(msg):
            return "text-recommend", {"query": msg}
        return "smalltalk", {"message": msg}

    def chat(self, user_msg: str) -> AgentReply:
        if has_ollama():
            tool, args = plan_with_llm(user_msg)
            mode = "ollama-agent"
        elif has_openai():
            tool, args = self._rule_plan(user_msg)
            mode = "openai-agent"
        else:
            tool, args = self._rule_plan(user_msg)
            mode = "agent-rule"

        if tool == "image-search":
            items = self.tools.search_image(args.get("image_url","").strip(), top_k=8)
            if not items:
                return AgentReply(mode=mode, message="That image didn’t match well. Describe the product and budget, and I’ll recommend options.", items=[])
            txt = self._render_list(items)
            return AgentReply(mode=mode, message=txt, items=items[:6])

        if tool == "text-recommend":
            items = self.tools.search_text(args.get("query",""), top_k=12)
            txt = self._render_list(items)
            return AgentReply(mode=mode, message=txt, items=items[:6])

        if tool == "qa-rag":
            msg, items = self.tools.qa_rag(args.get("question",""), hint_ids=None)
            return AgentReply(mode=mode, message=msg, items=items[:6] if items else [])

        s = self.tools.smalltalk(args.get("message",""))
        return AgentReply(mode=mode, message=s, items=[])

    def _render_list(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            return "No matching items in the catalog. Add a category or price, e.g., “running tee under $30”."
        lines = ["Here are some picks:"]
        for p in items[:6]:
            t = p.get("title","")
            b = p.get("brand","")
            c = p.get("category","")
            pr = f"${float(p.get('price',0)):.2f}"
            lines.append(f"- {t} — {pr} ({b} · {c})")
        return "\n".join(lines)
