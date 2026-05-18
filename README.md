# Indo Legal AI

AI Legal Assistant for Indonesian law — a RAG (Retrieval-Augmented Generation)
system that answers questions about Indonesian legislation with accurate,
verified citations.

This is **Phase 1 (V1)**: document ingestion, retrieval, and a cited Q&A endpoint.

## Tech Stack

- **Backend:** Python 3.12, FastAPI
- **Database / Vector store:** PostgreSQL + pgvector (one DB for vectors,
  document metadata, and chat history)
- **Embeddings:** HuggingFace multilingual (`intfloat/multilingual-e5-base`)
- **LLM:** provider abstraction — Ollama (local) or OpenAI (API), selected via `.env`
- **Deployment:** Docker + Docker Compose
- **Tracing:** LangSmith

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── routers/      # FastAPI route handlers
│   │   ├── services/     # ingestion, retrieval, generation, citation verifier
│   │   ├── models/       # SQLAlchemy models
│   │   ├── config.py     # settings loaded from environment
│   │   ├── database.py   # SQLAlchemy engine + session
│   │   └── main.py       # FastAPI app entrypoint
│   └── requirements.txt
├── frontend/
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── README.md
```

## Getting Started

1. Copy the environment template and adjust values as needed:

   ```sh
   cp .env.example .env
   ```

2. Build and start the stack:

   ```sh
   docker compose up --build
   ```

   If port 8000 is already in use on your machine, set `BACKEND_PORT` in
   `.env` to a free port before starting.

3. Verify the backend is healthy (replace 8000 with your `BACKEND_PORT`):

   ```sh
   curl http://localhost:8000/health
   ```

   Expected response:

   ```json
   {"status": "ok", "version": "0.1.0", "database": "up"}
   ```

   Interactive API docs are available at http://localhost:8000/docs.

## Build Status

Phase 1 is built in stages:

- [x] **Stage 1** — Project skeleton + Docker, `/health` endpoint
- [x] **Stage 2** — PDF ingestion pipeline (`/documents/*`)
- [x] **Stage 3** — Embedding + retrieval (pgvector cosine + reranker)
- [x] **Stage 4** — Chat endpoint with citations (`/chat/*`)
- [x] **Stage 5** — Citation verifier (existence + text-support + claim-grounding)
- [ ] **Stage 6** — Frontend
- [ ] **Stage 7** — LangSmith tracing
