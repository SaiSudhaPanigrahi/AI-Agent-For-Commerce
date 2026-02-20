# Mercury Architecture

## Goal

Mercury is a single-agent commerce assistant that supports:

- Natural language chat and recommendations
- Text search with category/color/price filters
- Image similarity search from upload or URL

The design emphasizes graceful degradation:

- If LLM keys are available, agent responses improve with tool calling.
- If LLM keys are missing, deterministic routing and retrieval still work.

## High-Level Components

- `backend/app.py`
  - FastAPI entrypoint
  - API routes, request validation, and error shaping
  - Request latency logging
- `backend/agent/agent.py`
  - Agent orchestration and intent handling
  - Small-talk, recommend, and image URL routing logic
- `backend/services/text_index.py`
  - Text retrieval and filtering
  - Sentence-transformers path with TF-IDF fallback
- `backend/services/vision_search.py`
  - Visual retrieval index and scoring
  - CLIP/ResNet/HSV fallback chain
- `backend/services/catalog_loader.py`
  - Catalog generation from `backend/data/*`
  - Filename-first color extraction (source of truth)
- `frontend/src/ui/App.tsx`
  - Unified UI for chat, text search, and image search

## Request Flow

1. Frontend calls one of:
   - `POST /api/chat`
   - `POST /api/search_text`
   - `POST /api/search_image`
   - `POST /api/search_by_url`
2. FastAPI validates request payloads via `backend/schemas.py`.
3. Route dispatches to agent/text/vision services.
4. Service returns ranked catalog items.
5. API returns typed JSON response to frontend.

## Data Model

Core catalog item fields:

- `id`
- `title`
- `category`
- `color`
- `price`
- `description`
- `image_path`
- optional `tags`, `score`

## Source of Truth for Catalog Data

- Product media lives in `backend/data/<category>/`.
- File names encode color intent (e.g., `bag8_pink.jpg`).
- Catalog is regenerated from folder contents and file names.

This avoids stale or hand-edited metadata drift and aligns UI color labels with file naming intent.

## Ranking Strategy

### Text Search

- Semantic similarity (or TF-IDF fallback)
- Strict filters for category/color/price
- Lightweight reranking with tag overlap and budget proximity boosts

### Image Search

- Embedding similarity
- Color bonus when detected color matches
- HSV histogram similarity
- Category prior from top visual neighbors

## Reliability and Operability

- Typed request/response schemas for stable API contracts
- Validation and consistent JSON errors
- Request logging with status and latency
- API tests with `FastAPI TestClient`
- Containerized local run via Docker Compose

## Tradeoffs

- Startup/index rebuild can be heavier for large catalogs.
- In-memory retrieval is simple and fast for small datasets, but not ideal for very large catalogs.
- Current architecture optimizes demo velocity and readability over distributed scalability.

## Evolution Path

- Move vectors to dedicated vector DB (`pgvector`/`Qdrant`)
- Add async background reindex pipeline
- Add metrics dashboard (latency, hit-rate, error rate)
- Add user/session personalization features
