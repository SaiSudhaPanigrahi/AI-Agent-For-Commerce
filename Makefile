.PHONY: up down logs test-backend reindex backend-dev frontend-dev

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

test-backend:
	cd backend && pytest tests/test_api.py

reindex:
	curl -sS -X POST http://localhost:8000/api/reindex | cat

backend-dev:
	cd backend && uvicorn app:app --host 0.0.0.0 --port 8000 --reload

frontend-dev:
	cd frontend && npm run dev
