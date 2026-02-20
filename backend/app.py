from __future__ import annotations
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List

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
    ErrorResponse,
    OperationResponse,
    RepairPathsResponse,
    SearchByUrlRequest,
    SearchFilters,
    SearchResultsResponse,
    SearchTextRequest,
    SearchTextResponse,
)
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

from services.path_repair import repair_paths

@app.post("/api/repair_paths", response_model=RepairPathsResponse)
def repair_paths_route():
    changed = repair_paths(catalog_path=catalog_path, data_dir=DATA_DIR)
    # rebuild indices so everything is in sync
    global text_index, vision_index
    text_index = TextIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, force_rebuild=True)
    vision_index = VisionIndex(catalog_path=catalog_path, cache_dir=CACHE_DIR, data_dir=DATA_DIR, force_rebuild=True)
    return RepairPathsResponse(ok=True, changed=changed)
