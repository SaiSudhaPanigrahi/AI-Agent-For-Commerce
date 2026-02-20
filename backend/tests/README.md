# Backend Test Suite

These tests use `FastAPI TestClient` with mocked backend services so they run fast and deterministically.

## Install

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
```

## Run

```bash
cd backend
pytest tests/test_api.py
```

## What is covered

- `GET /api/catalog`
- `POST /api/search_text` success + validation errors
- `POST /api/search_by_url` success + validation errors
- `POST /api/chat` blank-message guard
- `POST /api/reindex` success response
