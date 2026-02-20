from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, AnyHttpUrl, BaseModel, ConfigDict, Field


class CatalogItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    title: str
    category: str
    color: Optional[str] = None
    price: float
    description: str
    image_path: str = Field(
        validation_alias=AliasChoices("image_path", "image"),
        serialization_alias="image_path",
    )
    tags: List[str] = Field(default_factory=list)
    score: Optional[float] = None


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: Optional[str] = None
    color: Optional[str] = None


class SearchTextRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    q: str = Field(
        default="",
        validation_alias=AliasChoices("q", "query", "message", "text", "prompt"),
    )
    filters: SearchFilters = Field(default_factory=SearchFilters)
    category: Optional[str] = None
    color: Optional[str] = None
    min_price: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("min_price", "minPrice", "priceMin"),
    )
    max_price: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("max_price", "maxPrice", "priceMax"),
    )
    k: int = Field(default=12, ge=1, le=50, validation_alias=AliasChoices("k", "topK", "limit"))


class SearchTextResponse(BaseModel):
    results: List[CatalogItem]
    filters: SearchFilters
    q: str
    k: int


class SearchByUrlRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: AnyHttpUrl
    k: int = Field(default=8, ge=1, le=50)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    intent: str
    source: Optional[str] = None
    text: str
    reply: str
    results: List[CatalogItem] = Field(default_factory=list)
    filters: Dict[str, Optional[str]] = Field(default_factory=dict)


class SearchResultsResponse(BaseModel):
    results: List[CatalogItem]


class OperationResponse(BaseModel):
    ok: bool
    message: str


class RepairPathsResponse(BaseModel):
    ok: bool
    changed: int


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    details: Dict[str, Any] = Field(default_factory=dict)


class CompareAdviceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    category: str
    color: Optional[str] = None
    price: float
    description: str
    score: Optional[float] = None
    image_path: Optional[str] = None


class CompareAdviceRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: List[CompareAdviceItem] = Field(min_length=1, max_length=3)
    user_goal: Optional[str] = Field(default="Best overall value")
    require_ai: bool = False


class CompareAdviceResponse(BaseModel):
    summary: str
    bullets: List[str] = Field(default_factory=list)
    recommended_item_id: Optional[str] = None
    source: Literal["llm", "heuristic"] = "heuristic"
