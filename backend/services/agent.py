
from typing import List, Dict, Any
from .recommender import HybridRecommender
from .agent_llm import plan_with_llm, respond_with_llm

class AgentResult:
    def __init__(self, message: str, mode: str, items: List[Dict[str, Any]] | None = None):
        self.message = message
        self.mode = mode
        self.items = items

class MercuryAgent:
    def __init__(self, catalog: List[Dict[str, Any]]):
        self.catalog = catalog
        self.recommender = HybridRecommender(catalog)

    def recommend(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        return self.recommender.recommend(query, k=k)

    def chat(self, user_msg: str) -> AgentResult:
        tool, args, backend = plan_with_llm(user_msg)
        if tool == "recommend":
            q = args.get("query", user_msg)
            k = int(args.get("top_k", 8))
            items = self.recommend(q, k)
            ctx = "\n".join(f"- {p['title']} • {p['brand']} • {p.get('color','')} • ${p['price']}" for p in items[:8])
            answer = respond_with_llm(user_msg, ctx)
            return AgentResult(answer, mode=backend, items=items)
        else:
            answer = respond_with_llm(user_msg, "")
            return AgentResult(answer, mode=backend, items=None)
