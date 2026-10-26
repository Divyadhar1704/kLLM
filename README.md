# KrishiMitra Backend

AI agricultural assistant backend for the Krishindra application.
Text, image and video questions in English, Telugu, Kannada and Hindi, grounded in trusted agricultural sources (RAG).

This repository is the backend service only. The web UI under `app/web` is a developer testing client, not the farmer-facing frontend.

## Stack

FastAPI · Python 3.12 · PostgreSQL 16 + pgvector · Redis · Gemini on Vertex AI · Google Cloud Storage · Cloud Run (asia-south1)

## Layout

```
app/
  api/v1/        HTTP routes (versioned)
  core/          config, logging, security, errors
  models/        SQLAlchemy ORM models
  schemas/       Pydantic request/response schemas
  services/      business logic (chat, llm, embeddings, media)
  repositories/  database access
  rag/           ingestion, chunking, retrieval
  vision/        image quality gate, disease diagnosis
  video/         video validation, keyframe extraction
  worker/        background job entrypoints (Cloud Tasks)
  web/           developer test UI (disabled in production)
tests/           unit/ and integration/
scripts/         ingestion, db, ops scripts
documents/       source agricultural documents (raw/ is git-ignored)
evals/           golden Q&A evaluation sets (en/te/kn/hi)
```

## Local setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

Local infrastructure (needs Docker Desktop, used from Phase 9):

```powershell
docker compose up -d
docker compose ps
```

## Knowledge base ingestion (RAG)

Put agricultural documents (PDF, DOCX, TXT, MD) under `documents/raw/` (git-ignored), then:

```powershell
python -m scripts.ingest              # -> documents/processed/chunks.jsonl, manifest.json,
                                      #    pesticide_restrictions.json
python -m scripts.ingest --show 2     # also print sample chunks per document type
```

Unchanged files are served from `documents/processed/cache/`; any change to the ingestion code
re-processes everything automatically. Images are skipped (they feed the image model, Phase 16).

Then embed the chunks (resumable; vectors are cached in `documents/processed/embeddings.sqlite`):

```powershell
python -m scripts.embed               # free tier: ~90 chunks/min (EMBEDDING_REQUESTS_PER_MINUTE)
python -m scripts.embed --check       # multilingual retrieval sanity check over all vectors
```

Then load them into Postgres/pgvector (tables live in the private `krishimitra` schema):

```powershell
alembic upgrade head                  # create/upgrade tables (once per database)
python -m scripts.index               # sync DB = chunks.jsonl + cached embeddings
python -m scripts.index --stats
python -m scripts.index --search "టమాటాలో ఎర్లీ బ్లైట్ నివారణ" --crop tomato
```

Measure retrieval accuracy on the golden set (`evals/retrieval_v1.jsonl`, 4 languages):

```powershell
python -m scripts.eval_retrieval --show-misses     # hybrid vs vector-only: hit@1, hit@5, MRR
```

Ask the running API (prints sources, grounding, safety warnings and per-stage timings):

```powershell
python -m scripts.ask --demo
python -m scripts.ask "పత్తిలో తెల్లదోమ నివారణ ఎలా?"
```

## System testing (against a running server)

```powershell
uvicorn app.main:app --port 8010                 # terminal 1
python -m scripts.system_test                    # edge cases, errors, security, concurrency
python -m scripts.system_test --no-ai            # same, without AI calls (no quota used)
python -m scripts.eval_answers                   # 20-question answer-quality eval, 4 languages
python -m scripts.eval_retrieval                 # retrieval accuracy (hit@k, MRR)
```

Database and live-AI tests (skipped by default):

```powershell
$env:KRISHIMITRA_DB_TESTS="1"; $env:KRISHIMITRA_RUN_EXTERNAL="1"; pytest tests/integration tests/external
pytest --cov=app                                  # coverage
```

## Quality checks

```powershell
ruff check .
ruff format --check .
mypy
pytest
```

## Dependency management

Add top-level dependencies to `requirements.in` or `requirements-dev.in`, then regenerate the locks:

```powershell
pip-compile --generate-hashes requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt
pip-sync requirements.txt requirements-dev.txt
```
