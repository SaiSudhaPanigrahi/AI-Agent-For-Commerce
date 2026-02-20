# Contributing Guide

Thanks for your interest in improving Mercury.

## Prerequisites

- Python 3.11+ (3.12 recommended)
- Node.js 18+
- Optional: Docker + Docker Compose

## Local Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Full Stack via Docker

```bash
make up
```

## Tests

Backend API tests:

```bash
make test-backend
```

## Development Rules

- Keep API contracts typed through `backend/schemas.py`.
- Preserve backward compatibility for existing frontend route usage.
- Prefer deterministic behavior for catalog generation and indexing.
- Keep filename-derived product metadata as source truth unless explicitly changing data model strategy.

## Pull Request Checklist

- Include a clear problem statement and change summary.
- Add or update tests for behavior changes.
- Document API/interface changes in `README.md` if needed.
- For non-trivial changes, include notes on tradeoffs and rollback plan.

## Issue Reporting

Please include:

- Repro steps
- Expected behavior
- Actual behavior
- Logs or API responses when available

## Roadmap Contributions

High-impact areas:

- Search quality evaluation and benchmarking
- Observability and metrics
- Vector database integration
- Frontend UX improvements for result explainability
