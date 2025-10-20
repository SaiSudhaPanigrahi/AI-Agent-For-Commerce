from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Product(BaseModel):
    id: str
    title: str
    category: str
    description: str
    price: float
    image_url: str
    brand: Optional[str] = None
    tags: Optional[List[str]] = None

class ChatRequest(BaseModel):
    user_id: str = Field(default="demo")
    message: str

class ChatResponse(BaseModel):
    reply: str
    mode: Literal["local-lite","openai","ollama-agent"]
    items: Optional[List[Product]] = None

class RecommendRequest(BaseModel):
    user_id: str = Field(default="demo")
    query: str
    top_k: int = 8

class RecommendResponse(BaseModel):
    items: List[Product]

class ImageSearchResponse(BaseModel):
    items: List[Product]

class CatalogResponse(BaseModel):
    items: List[Product]
