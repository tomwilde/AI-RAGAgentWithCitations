# RAG Agent with Citations

Streamlit app that ingests a PDF, creates embeddings with semantic-ish chunking, stores vectors in Qdrant, and answers questions with confidence + numbered citations.

## Prerequisites

- Docker + Docker Compose
- OpenRouter API key in your shell: `export OPENROUTER_API_KEY=...`

## Run (App Profile)

```bash
docker compose --profile app up --build
```

Then open `http://localhost:8501`.

## Run (Debug Profile)

```bash
docker compose --profile debug up --build
```

- Streamlit app: `http://localhost:8501`
- Debugpy attach: `localhost:5678`
- In debug mode only `app.py` and `src/` are bind-mounted for live edits.
- This avoids container writes corrupting host `.venv` ownership.

## Stop

```bash
docker compose down
```

## Make Targets

```bash
make app-start
make app-stop
make debug-start
make debug-stop
```

- `app-start` / `debug-start` perform preflight checks to ensure prior compose services are shut down and required host ports are free.

## Local QA

```bash
uv sync
.venv/bin/pytest tests/ -v
.venv/bin/ruff check .
.venv/bin/ty check
```

## Notes

- `OPENROUTER_API_KEY` is read from environment only and never persisted.
- Upload a PDF to automatically start embedding generation, then query using **Query**.
- Embedding generation runs as a single Streamlit-controlled process with progress updates.
- Each new embedding run clears previous vectors before indexing the new PDF.
- Embedding generation uses parallel workers (configure via `EMBEDDING_WORKERS`, default `6`).
- Streamlit upload/message limits are set to 10GB in `.streamlit/config.toml`.
- The UI includes a `Use Agentic RAG` toggle to compare simple vs agentic behavior.
- Agentic mode uses two gates: retrieval score gate (`<0.15` insufficient local context, `0.15-0.30` low confidence generation) and model self-verification (`supported`).
- If local retrieval is not sufficient, it fails gracefully with a user-friendly message (offline-safe, no web fallback).
