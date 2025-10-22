
from typing import List, Optional, Literal
from pydantic import BaseModel

class Product(BaseModel):
    id: str
    title: str
    brand: str
    category: str
    color: Optional[str] = None
    price: float
    description: str
    image: str  # URL path like /images/bag1.jpg

class CatalogResponse(BaseModel):
    items: List[Product]

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 8

class RecommendResponse(BaseModel):
    items: List[Product]

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    mode: Literal["agent", "ollama"] = "ollama"
    message: str
    items: Optional[List[Product]] = None
