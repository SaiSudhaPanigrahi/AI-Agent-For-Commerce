# Mercury — AI Agent for a Commerce Website (Polished, No Docker)

**One agent, three modes:** general chat · text recommendations · image-based product search.  
Built for quick local dev and a clean presentation: **FastAPI backend** + **Vite React (TS)** + **Tailwind**. Optional OpenAI chat.

---

## ✅ Requirements Coverage

- **User-friendly frontend interface** → Polished React UI with a hero, badges, chat, image search, recs, and full catalog.
- **Documented agent API** → Swagger at **`http://localhost:8000/docs`** (FastAPI auto-docs) + `GET /health`.
- **Single agent handles all** → One backend app exposes **chat**, **text-based recs**, and **image-based search** against the same catalog.
- **Catalog-limited** → Both recommendation and visual search operate only over `backend/data/catalog.json`.
- **Tech choices explained** → See **Design** below.
- **Optional LLM** → Works offline with local smalltalk; if `OPENAI_API_KEY` is set, chat upgrades automatically.

---

## 🚀 Quickstart

### 1) Backend (FastAPI)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# API docs → http://localhost:8000/docs
# Health → http://localhost:8000/health
```

*(Optional)* richer chat via OpenAI:
```bash
export OPENAI_API_KEY=sk-...          # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
export OPENAI_MODEL=gpt-4o-mini
uvicorn app:app --reload --port 8000
```

*(Optional)* pre-download model weights to speed first request:
```bash
python scripts/prewarm.py
```

### 2) Frontend (Vite + React + Tailwind)
```bash
cd ../frontend
cp .env.example .env                  # VITE_API_BASE=http://localhost:8000
npm install
npm run dev                           # UI → http://localhost:5173
```

---

## 🧠 Agent Behavior

- **Chat (`POST /api/chat`)**  
  - If `OPENAI_API_KEY` is set → uses OpenAI with a neat system prompt.  
  - Otherwise → `local-lite` smalltalk that clearly advertises capabilities.

- **Text-based Recommendations (`POST /api/recommend`)**  
  - Build a text corpus: `title | brand | category | description | tags`.  
  - Encode with **MiniLM (all-MiniLM-L6-v2)**; cosine top‑k retrieval.

- **Image-based Product Search (`POST /api/image-search`)**  
  - Encode catalog images & query image using **CLIP (ViT-B/32)**; cosine top‑k neighbors.

- **Catalog (`GET /api/catalog`)** → returns the items used by both recommenders.

- **Health (`GET /health`)** → sanity check with item count.

---

## 🧪 API Examples

```bash
# Chat
curl -X POST http://localhost:8000/api/chat   -H "Content-Type: application/json"   -d '{"user_id":"demo","message":"What can you do?"}'

# Text recs
curl -X POST http://localhost:8000/api/recommend   -H "Content-Type: application/json"   -d '{"user_id":"demo","query":"lightweight sports tee under $30","top_k":8}'

# Image search
curl -X POST http://localhost:8000/api/image-search   -F "image_url=https://images.unsplash.com/photo-1516826957135-700dedea698c?q=80&w=800"

# Catalog
curl http://localhost:8000/api/catalog
```

---

## 🧱 Design

- **Unification**: one FastAPI app orchestrates chat + text recs + visual search for a consistent agent feel.
- **Embeddings-first**: strong baseline without custom training; easy to swap in a vector DB later.
- **Optional LLM**: graceful degradation; reviewers can run it offline.
- **Frontend UX**: clean, modern styling with hero, badges, and helpful defaults.
- **Maintainability**: thin services layer (`services/*`), typed schemas, and a clear catalog boundary.

**Future work** (if time allowed): cross-encoder reranking, pgvector/Qdrant, click tracking + analytics, admin panel for catalog and embeddings refresh.

---

## 🗂️ Repo Layout

```
ai-commerce-agent-pro/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── schemas.py
│   ├── data/
│   │   └── catalog.json
│   ├── scripts/
│   │   └── prewarm.py
│   └── services/
│       ├── chat.py
│       ├── image_search.py
│       ├── recommender.py
│       └── utils.py
└── frontend/
    ├── .env.example
    ├── index.html
    ├── package.json
    ├── postcss.config.js
    ├── tailwind.config.js
    ├── tsconfig.json
    ├── vite.config.ts
    └── src/
        ├── App.tsx
        ├── index.css
        ├── main.tsx
        ├── lib/api.ts
        └── components/ProductCard.tsx
```

---

## ⚠️ Notes

- First run downloads model weights; `scripts/prewarm.py` shortens the wait.
- Everything uses CPU by default.
- Image-based search fetches remote images by URL; ensure you paste a valid public image link.
