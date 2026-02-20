from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette import status

from dotenv import load_dotenv

from schemas import (
    CatalogItem,
    ChatRequest,
    ChatResponse,
    CompareAdviceRequest,
    CompareAdviceResponse,
    ErrorResponse,
    HealthResponse,
    OperationResponse,
    RepairPathsResponse,
    SearchByUrlRequest,
    SearchFilters,
    SearchResultsResponse,
    SearchTextRequest,
    SearchTextResponse,
)
from agent.gemini_client import get_gemini_smalltalk
from agent.agent import Agent
from services.catalog_loader import ensure_catalog
from services.text_index import TextIndex
from services.vision_search import VisionIndex



APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
CACHE_DIR = APP_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load .env early so child processes see it too
load_dotenv(dotenv_path=APP_DIR / ".env", override=False)  # <-- NEW

app = FastAPI(title="AI Commerce Agent")
APP_STARTED_AT = time.time()
logger = logging.getLogger("ai_commerce_agent")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "request_failed method=%s path=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request_complete method=%s path=%s status=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(error="Validation error", detail=exc.errors()).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    detail = exc.detail if exc.detail is not None else "Request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="Request failed", detail=detail).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("unhandled_error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="Internal server error").model_dump(),
    )

# replace your existing mount with this:
app.mount(
    "/data",
    StaticFiles(directory=str(DATA_DIR), html=False, check_dir=True),
    name="data",
)


# Bootstrap
# Source of truth is backend/data/* filenames; regenerate catalog on startup.
catalog_path = ensure_catalog(DATA_DIR, CACHE_DIR, regenerate=True)
text_index = TextIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, force_rebuild=False)
vision_index = VisionIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, data_dir=DATA_DIR, force_rebuild=False)
agent = Agent(text_index=text_index, vision_index=vision_index)


@app.get("/healthz", response_model=HealthResponse)
def healthz():
    uptime_seconds = round(time.time() - APP_STARTED_AT, 3)
    return HealthResponse(
        status="ok",
        service="ai-commerce-agent",
        details={
            "uptime_seconds": uptime_seconds,
        },
    )


@app.get("/readyz", response_model=HealthResponse)
def readyz():
    checks = {
        "catalog_path_exists": catalog_path.exists(),
        "catalog_loaded": bool(getattr(text_index, "catalog", None)),
        "text_index_ready": hasattr(text_index, "search_with_filters"),
        "vision_index_ready": hasattr(vision_index, "search_image_url"),
    }
    ready = all(checks.values())
    payload = HealthResponse(
        status="ok" if ready else "degraded",
        service="ai-commerce-agent",
        details={
            **checks,
            "catalog_items": len(getattr(text_index, "catalog", [])),
        },
    )
    if not ready:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump())
    return payload

@app.get("/api/catalog", response_model=List[CatalogItem])
def get_catalog():
    return text_index.catalog

# @app.post("/api/reindex")
# def reindex():
#     global text_index, vision_index, agent
#     cat = ensure_catalog(DATA_DIR, CACHE_DIR, regenerate=True)
#     text_index = TextIndex(catalog_path=cat, cache_dir=CACHE_DIR, force_rebuild=True)
#     vision_index = VisionIndex(catalog_path=cat, cache_dir=CACHE_DIR, data_dir=DATA_DIR, force_rebuild=True)
#     agent = Agent(text_index=text_index, vision_index=vision_index)
#     return {"ok": True}

# --- REINDEX: rebuild indexes ONLY, keep your edited catalog.json as-is ---
@app.post("/api/reindex", response_model=OperationResponse)
def reindex_only():
    global text_index, vision_index, catalog_path
    # Source of truth is filesystem under backend/data/*.
    # Regenerate catalog first, then rebuild indices.
    catalog_path = ensure_catalog(DATA_DIR, CACHE_DIR, regenerate=True)
    text_index = TextIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, force_rebuild=True)
    vision_index = VisionIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, data_dir=DATA_DIR, force_rebuild=True)
    return OperationResponse(ok=True, message="Regenerated catalog from data folders and rebuilt indexes.")

# --- REBUILD CATALOG: optional, will overwrite manual edits from folders ---
@app.post("/api/rebuild_catalog", response_model=OperationResponse)
def rebuild_catalog_and_indexes():
    global text_index, vision_index, catalog_path
    # This one regenerates catalog.json from the data/* folders
    from services.catalog_loader import ensure_catalog
    catalog_path = ensure_catalog(DATA_DIR, CACHE_DIR, regenerate=True)  # <-- overwrites
    text_index = TextIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, force_rebuild=True)
    vision_index = VisionIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, data_dir=DATA_DIR, force_rebuild=True)
    return OperationResponse(ok=True, message="Regenerated catalog.json from folders and rebuilt indexes.")


@app.post("/api/search_text", response_model=SearchTextResponse)
async def search_text(req: SearchTextRequest):
    category = req.category or req.filters.category
    color = req.color or req.filters.color
    min_price = req.min_price
    max_price = req.max_price
    q = req.q or ""
    k = req.k

    if min_price is not None and max_price is not None and min_price > max_price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_price must be less than or equal to max_price",
        )

    items = text_index.search_with_filters(
        q,
        category=category,
        color=color,
        min_price=min_price,
        max_price=max_price,
        top_k=k,
    )

    return SearchTextResponse(
        results=items,
        filters=SearchFilters(category=category, color=color),
        q=q,
        k=k,
    )

@app.post("/api/search_image", response_model=SearchResultsResponse)
async def search_image(file: UploadFile = File(...), k: int = Form(default=8, ge=1, le=50)):
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    suffix = os.path.splitext(file.filename or "")[-1] or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(content)
    try:
        items = vision_index.search_image_path(Path(tmp_path), top_k=k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to process uploaded image",
        ) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return SearchResultsResponse(results=items)


@app.post("/api/search_by_url", response_model=SearchResultsResponse)
async def search_by_url(req: SearchByUrlRequest):
    items = vision_index.search_image_url(str(req.url), top_k=req.k)
    return SearchResultsResponse(results=items)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must not be blank",
        )
    plan = await agent.chat(message)
    return plan


def _score_for_compare(item) -> float:
    raw = item.score
    if raw is None:
        return 0.55
    return max(0.0, min(1.0, (float(raw) + 1.0) / 2.0))


def _heuristic_compare_advice(req: CompareAdviceRequest) -> CompareAdviceResponse:
    items = req.items
    prices = [float(it.price) for it in items]
    p_min = min(prices)
    p_max = max(prices)
    spread = max(1e-6, p_max - p_min)

    scored = []
    for it in items:
        score = _score_for_compare(it)
        affordability = 1.0 - ((float(it.price) - p_min) / spread)
        value = 0.6 * score + 0.4 * affordability
        scored.append((it, score, affordability, value))

    best_value = max(scored, key=lambda x: x[3])
    best_budget = min(scored, key=lambda x: float(x[0].price))
    best_quality = max(scored, key=lambda x: x[1])

    summary = f"Best overall choice: {best_value[0].title}."
    bullets = [
        f"Best overall value: {best_value[0].title} (value {best_value[3]:.2f}, price ${float(best_value[0].price):.2f}).",
        f"Best budget option: {best_budget[0].title} at ${float(best_budget[0].price):.2f}.",
        f"Best quality/relevance: {best_quality[0].title} (score {best_quality[1]:.2f}).",
    ]
    if req.user_goal:
        bullets.append(f"Goal considered: {req.user_goal}.")

    return CompareAdviceResponse(
        summary=summary,
        bullets=bullets,
        recommended_item_id=best_value[0].id,
        source="heuristic",
    )


def _clean_ai_text_line(text: str) -> str:
    line = (text or "").strip()
    if not line:
        return ""
    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = line.lstrip("-•* ").strip()
    line = re.sub(r"^\d+\.\s*", "", line)
    return line.strip()


def _extract_json_object(text: str) -> Optional[dict]:
    raw = (text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    candidates.extend(fenced)

    for candidate in candidates:
        body = candidate.strip()
        if not body:
            continue
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _parse_compare_ai_output(text: str, valid_ids: set[str], fallback_id: Optional[str]) -> Tuple[str, List[str], Optional[str]]:
    parsed = _extract_json_object(text)
    if parsed is not None:
        summary = _clean_ai_text_line(str(parsed.get("summary") or ""))
        bullets_raw = parsed.get("bullets") or []
        bullets = []
        if isinstance(bullets_raw, list):
            bullets = [_clean_ai_text_line(str(x)) for x in bullets_raw]
            bullets = [b for b in bullets if b]
        winner = str(parsed.get("winner_id") or parsed.get("recommended_item_id") or "").strip()
        if winner and winner not in valid_ids:
            winner = ""
        winner_id = winner or fallback_id
        if not summary and bullets:
            summary = bullets[0]
        return summary, bullets[:4], winner_id

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    summary = ""
    bullets: List[str] = []
    for ln in lines:
        cleaned = _clean_ai_text_line(ln)
        if not cleaned:
            continue
        if ln.lstrip().startswith(("-", "•", "*")) or re.match(r"^\d+\.", ln.strip()):
            bullets.append(cleaned)
        elif not summary:
            summary = cleaned
        else:
            bullets.append(cleaned)

    if not summary and bullets:
        summary = bullets[0]
    return summary, bullets[:4], fallback_id


@app.post("/api/compare_advice", response_model=CompareAdviceResponse)
async def compare_advice(req: CompareAdviceRequest):
    # Reload backend/.env so local key updates apply without restarting the server process.
    load_dotenv(dotenv_path=APP_DIR / ".env", override=True)
    heuristic = _heuristic_compare_advice(req)

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        logger.warning("compare_advice_fallback: GOOGLE_API_KEY missing")
        return heuristic

    try:
        model = get_gemini_smalltalk()
        compact = [
            {
                "id": it.id,
                "title": it.title,
                "category": it.category,
                "color": it.color,
                "price": float(it.price),
                "description": it.description,
            }
            for it in req.items
        ]
        prompt = (
            "You are an e-commerce comparison assistant. "
            "Choose the best overall option and provide detailed, practical rationale.\n\n"
            f"User goal: {req.user_goal}\n"
            f"Items: {compact}\n\n"
            "Return STRICT JSON only (no markdown, no extra text) using this schema:\n"
            "{\n"
            '  "winner_id": "<item id>",\n'
            '  "summary": "<one sentence, 18-30 words>",\n'
            '  "bullets": ["...", "...", "...", "..."]\n'
            "}\n"
            "Rules:\n"
            "- Exactly 4 bullets.\n"
            "- Each bullet must be 16-28 words.\n"
            "- Mention concrete tradeoffs (price, category/use case, color/style, description-based utility).\n"
            "- Compare at least two items by title in each bullet.\n"
            "- Do not use markdown symbols."
        )
        resp = await asyncio.to_thread(model.generate_content, [prompt])
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            valid_ids = {it.id for it in req.items}
            summary, bullets, winner_id = _parse_compare_ai_output(
                text=text,
                valid_ids=valid_ids,
                fallback_id=heuristic.recommended_item_id,
            )
            if len(summary) < 20 or len(bullets) < 3:
                refine_prompt = (
                    "Rewrite the following compare output into richer STRICT JSON.\n"
                    f"Items: {compact}\n"
                    f"Raw output: {text}\n\n"
                    "Return JSON only with winner_id, summary, bullets[4]. "
                    "Keep summary 18-30 words and each bullet 16-28 words with practical tradeoffs."
                )
                refine_resp = await asyncio.to_thread(model.generate_content, [refine_prompt])
                refine_text = (getattr(refine_resp, "text", None) or "").strip()
                if refine_text:
                    refined_summary, refined_bullets, refined_winner_id = _parse_compare_ai_output(
                        text=refine_text,
                        valid_ids=valid_ids,
                        fallback_id=winner_id,
                    )
                    if refined_summary:
                        summary = refined_summary
                    if len(refined_bullets) >= len(bullets):
                        bullets = refined_bullets
                    winner_id = refined_winner_id

            return CompareAdviceResponse(
                summary=summary or heuristic.summary,
                bullets=bullets[:4] if bullets else heuristic.bullets,
                recommended_item_id=winner_id or heuristic.recommended_item_id,
                source="llm",
            )
        logger.warning("compare_advice_fallback: empty llm response")
    except Exception as exc:
        logger.warning("compare_advice_llm_fallback: %s", exc)

    return heuristic

from services.path_repair import repair_paths

@app.post("/api/repair_paths", response_model=RepairPathsResponse)
def repair_paths_route():
    changed = repair_paths(catalog_path=catalog_path, data_dir=DATA_DIR)
    # rebuild indices so everything is in sync
    global text_index, vision_index
    text_index = TextIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, force_rebuild=True)
    vision_index = VisionIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, data_dir=DATA_DIR, force_rebuild=True)
    return RepairPathsResponse(ok=True, changed=changed)
