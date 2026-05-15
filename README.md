---
title: Steuer RAG
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Steuer-RAG — Bilingual (DE/EN) RAG for the German Steuererklärung

> **Purpose:** Help users in Germany answer questions about their *Steuererklärung*
> (income tax return) using only official primary sources:
> [`bundesfinanzministerium.de`](https://www.bundesfinanzministerium.de) (BMF),
> [`elster.de`](https://www.elster.de),
> [`bzst.de`](https://www.bzst.de), and
> [`gesetze-im-internet.de`](https://www.gesetze-im-internet.de) (EStG · AO · EStDV).
> **Stack:** Python ≥ 3.11 · LangChain · Chroma (vector) · BGE-M3 (multilingual embeddings) ·
> BGE-reranker-v2-m3 (cross-encoder) · Anthropic Claude or OpenAI (generation).

This repository is intentionally self-contained — every model, secret, dependency, and command
needed to rebuild the system is in here.

---

## Table of Contents

1. [System Scope & North Star](#1-system-scope--north-star)
2. [RAG System Architecture](#2-rag-system-architecture)
3. [Database Setup & Processing](#3-database-setup--processing)
4. [Models Used](#4-models-used)
5. [Configuration & Keys](#5-configuration--keys)
6. [Dependencies & Infrastructure](#6-dependencies--infrastructure)
7. [Reproducibility Guide (from scratch)](#7-reproducibility-guide-from-scratch)
8. [Operational Runbook](#8-operational-runbook)
9. [Appendix — Source Catalog](#9-appendix--source-catalog)

---

## 1. System Scope & North Star

Steuer-RAG is a **bilingual retrieval-augmented question-answering system** for the German
tax return. It targets two audiences:

- **German-speaking residents** asking in German (`Wie trage ich Werbungskosten ein?`)
- **English-speaking residents** in Germany (`How do I declare home-office expenses?`)

In both cases the same underlying corpus is queried; language selection drives prompt
rendering and a soft preference in retrieval ordering.

```
┌────────────────────────────────────┐
│  Scrapers (BMF, Elster, BZSt,      │── DocumentCore rows
│  Gesetze) + PDF parser             │
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│  Chunker (deterministic,           │── DocumentChunk rows
│  char-based, v1)                   │
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│  BGE-M3 embedder                   │── 1024-d vectors
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│  Chroma (vector + BM25             │── persistent kb
│  built in-process)                 │
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│  Hybrid retriever + BGE            │── top-k passages
│  cross-encoder reranker            │
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│  LCEL RAG chain                    │── cited answer (DE or EN)
│  (Anthropic / OpenAI)              │
└────────────────────────────────────┘
```

The pipeline is **deterministic by design**: content-addressable doc/chunk IDs, byte-stable chunk
offsets, pinned models, frozen `chunk_strategy_version = "v1"`.

---

## 2. RAG System Architecture

### 2.1 End-to-end data flow

```
External Sources
  ├── bundesfinanzministerium.de  (BMF-Schreiben, brochures, FAQ — DE + EN)
  ├── elster.de                   (public help / FAQ / forms listing — DE)
  ├── bzst.de                     (IdNr, ELStAM, Altersvorsorge, Kapitalerträge — DE + EN)
  └── gesetze-im-internet.de      (EStG, AO, EStDV — DE)
         │
         ▼
Async scrapers (steuer_rag/sources/*)
  · respect robots.txt
  · polite delay + bounded concurrency
  · on-disk byte cache (./data/raw/<source>/<sha>.bin)
  · depth-N BFS with thin-page filtering (< thin_html_chars → recurse only)
  · separate PDF discovery pipeline (pdf_allow_pattern per source)
         │
         ▼
schema/models.DocumentCore.build()
  · normalize_text()  · detect_language()  · content-addressable doc_id
         │
         ▼
schema/chunking.chunk_text()
  · chunk_size=1200, overlap=200, min=200, look_back=200
  · prefers \n\n > . > \n > space breaks
         │
         ▼
pipeline/embed.get_embeddings()   →  BAAI/bge-m3 (1024-d, multilingual)
         │
         ▼
pipeline/index.VectorIndex (Chroma persistent at ./data/chroma)
  · upsert by chunk_id  · metadata: source, doc_type, language, url, title, offsets
  · batched writes (4 000/batch) to stay under Chroma's hard cap
         │
         ▼
retrieval/search.HybridRetriever
  · dense: Chroma cosine
  · sparse: rank_bm25 (built from indexed docs)
  · fusion: LangChain EnsembleRetriever (RRF, weights 0.6 / 0.4)
  · rerank: BAAI/bge-reranker-v2-m3 (top_n=50, returns k)
         │
         ▼
generation/chain.build_rag_chain()   →   LCEL Runnable
  · prompts/get_prompt(lang)  · llm/get_llm()
  · returns RAGAnswer{ question, answer, language, citations[], passages[] }
```

### 2.2 Lane 1 — Ingestion (`steuer_rag/sources/`)

Every source extends `BaseScraper` (`sources/base.py`) which provides:

| Concern | How it's handled |
|--------|------------------|
| Politeness | `User-Agent` header from `STEUER_RAG_USER_AGENT`; `STEUER_RAG_REQUEST_DELAY_MS` between requests; `STEUER_RAG_MAX_CONCURRENCY` cap |
| robots.txt | `urllib.robotparser` — disallowed URLs are skipped with a log line |
| Retry | `tenacity` exponential backoff (3 attempts) on `TransportError` / 5xx / 429; 4xx errors are not retried |
| Caching | First fetch saves raw bytes to `./data/raw/<source>/<sha>.bin`; re-runs read from cache |
| HTML cleanup | `trafilatura` first (boilerplate-aware), `BeautifulSoup` fallback |
| PDF parsing | `pdfplumber` first (with `/Title` metadata extraction), `pypdf` fallback |
| Discovery | BFS from seed pages; HTML links filtered by `allow_pattern`; PDFs by `pdf_allow_pattern` |
| Depth | `max_depth` controls hops from seeds; thin nav pages (< `thin_html_chars`) are recursed but not indexed |
| Caps | `max_pages` per source — safety guard against runaway crawls |

Concrete scrapers:

| File | Source | Scope |
|------|--------|-------|
| `sources/bmf.py` | `bundesfinanzministerium` | BMF-Schreiben, brochures, FAQ — individual income tax only; excludes VAT, corporate, fiscal-stats |
| `sources/elster.py` | `elster` | Public help, FAQ, forms listing (most ELSTER is auth-walled) |
| `sources/bzst.py` | `bzst` | Privatpersonen + EN mirror; excludes fire-protection insurance, bank-side templates |
| `sources/gesetze.py` | `gesetze` | EStG + AO + EStDV section pages from gesetze-im-internet.de |

### 2.3 Lane 2 — Chunking (`steuer_rag/schema/chunking.py`)

**Character-based, deterministic, model-agnostic:**

```python
chunk_text(
    text,
    chunk_size=1200,        # target chars per chunk
    overlap=200,            # carry-over to preserve cross-chunk context
    min_chunk_chars=200,    # drop fragments smaller than this
    look_back=200,          # window in which we search for a natural break
)
```

- Line endings normalized to `\n` so offsets are stable across OS.
- Break-preference order: `\n\n` > `. ` > `\n` > ` `.
- `chunk_id = sha256(doc_id || chunk_index || content_hash)[:16]` — re-runs are idempotent.
- `chunk_strategy_version = "v1"` records the regime for future re-chunking sweeps.

### 2.4 Lane 3 — Embedding (`steuer_rag/pipeline/embed.py`)

- **Model:** `BAAI/bge-m3` (default; override via `STEUER_RAG_EMBED_MODEL`).
- **Why this model?** Multilingual (DE + EN + ~100 langs), 1024-d, Apache-2.0 license.
- **Provider:** `sentence-transformers` via `langchain_huggingface.HuggingFaceEmbeddings`.
- **Device:** `STEUER_RAG_EMBED_DEVICE` (cpu / cuda / mps), auto-batched at 32.

### 2.5 Lane 4 — Vector Store (`steuer_rag/pipeline/index.py`)

| Field | Value |
|-------|-------|
| Backend | Chroma (persistent on disk) |
| Path | `./data/chroma/` |
| Collection | `steuer_chunks` |
| Distance | cosine (`hnsw:space=cosine`) |
| Upsert | by `chunk_id` (re-running ingest is idempotent) |
| Write batch | 4 000 chunks/call (Chroma hard cap is ~5 461) |
| Metadata | flat scalars only — Chroma rejects nested types |

### 2.6 Lane 5 — Retrieval (`steuer_rag/retrieval/search.py`)

| Strategy | Underlying | Notes |
|----------|------------|-------|
| `dense` | Chroma similarity | Cosine, pure vector |
| `sparse` | `rank_bm25` via `BM25Retriever` | Built in-process from indexed chunks |
| `hybrid` | `EnsembleRetriever([dense, sparse], weights=[0.6, 0.4])` | RRF fusion |
| `hybrid_rerank` *(default)* | `hybrid` → BGE cross-encoder | Two-stage: 50 candidates → top-k by relevance |

Common features:

- `source_filter` (`bundesfinanzministerium` | `elster` | `bzst` | `gesetze`) — applied to Chroma metadata and BM25 post-filter.
- `language_preference` — soft preference (preferred-language docs sort first; the other language is never hard-dropped).
- Auto-detect: when `language` is `None`, inferred from the query.

### 2.7 Lane 6 — Generation (`steuer_rag/generation/`)

- **LCEL chain** — composes with LangChain callbacks, streaming, tracing.
- **Bilingual prompts** (`prompts.py`) — system + user templates for DE and EN: cite by `[n]`, refuse to invent, surface "context not sufficient" rather than hallucinate, end with a `Sources:` / `Quellen:` block.
- **LLM provider** — Anthropic Claude (default) or OpenAI, switched by env.

The chain returns a `RAGAnswer` dataclass:

```python
RAGAnswer(
    question: str,
    answer: str,
    language: Language,
    citations: list[Citation],  # n, source, url, title, rerank_score
    passages: list[Document],   # raw retrieved chunks
)
```

---

## 3. Database Setup & Processing

### 3.1 Storage layout

```
data/
  raw/<source>/<sha>.bin     # byte cache of every fetched URL
  chroma/                    # Chroma persistent collection
```

### 3.2 `steuer_chunks` collection schema

| Key | Type | Used for |
|-----|------|----------|
| `chunk_id` | str | primary key (sha prefix) |
| `doc_id` | str | join back to source doc |
| `document_key` | str | URL (stable cross-version key) |
| `source` | str | `bundesfinanzministerium` \| `elster` \| `bzst` \| `gesetze` |
| `doc_type` | str | `html` \| `pdf` |
| `url` | str | citation |
| `doc_title` | str | display |
| `section` | str | optional sub-section |
| `language` | str | `de` \| `en` \| `unknown` |
| `chunk_index`, `start_char`, `end_char`, `content_chars` | int | provenance |
| `content_hash` | str | idempotency |
| `chunk_strategy_version` | str (`"v1"`) | re-chunk marker |
| `created_at` | iso datetime | recency |
| *(content)* | text | embedded + BM25 input |
| *(vector)* | 1024-d float[] | dense search |

### 3.3 Approximate corpus size (full ingest)

| Source | Docs | Chunks | Content |
|--------|------|--------|---------|
| BMF | ~500 | ~17 000 | BMF-Schreiben (Einkommensteuer, Lohnsteuer, Kapitalertragsteuer), brochures, FAQ |
| Elster | ~112 | ~260 | Public help pages, forms listing |
| BZSt | ~65 | ~440 | IdNr, ELStAM, Altersvorsorge, Kapitalerträge, Kirchensteuer |
| Gesetze | ~700 | ~6 000 | EStG + AO section pages (EStDV in progress) |
| **Total** | **~1 380** | **~24 000** | |

### 3.4 Re-runs / refresh

- **Idempotent**: re-running `steuer-rag ingest <source>` writes the same chunk IDs → Chroma upserts in place. Safe to run on a schedule.
- **Hard reset**: delete `./data/chroma/` and `./data/raw/`, then re-ingest.
- **Model upgrade** (different embed model): delete `./data/chroma/` (vector dim/space is not portable across models), then re-ingest.

---

## 4. Models Used

| Role | Model | Provider | Where set | Dim / Cost |
|------|-------|----------|-----------|------------|
| Embedding | `BAAI/bge-m3` | HF / sentence-transformers | `STEUER_RAG_EMBED_MODEL` | 1024-d, local |
| Reranker | `BAAI/bge-reranker-v2-m3` | HF / sentence-transformers `CrossEncoder` | `STEUER_RAG_RERANK_MODEL` | top-50 → top-k |
| LLM (default) | `claude-sonnet-4-6` | Anthropic | `STEUER_RAG_LLM_MODEL` | API |
| LLM (alt) | `gpt-4o-mini` etc. | OpenAI | `STEUER_RAG_LLM_PROVIDER=openai` + `STEUER_RAG_LLM_MODEL` | API |

All three OSS models are multilingual — DE and EN queries land in the same space.

> **Intel Mac note:** Tested on Intel macOS. Pin `torch>=2.2,<2.6`, `transformers<5`,
> `sentence-transformers<4`, `numpy<2` — wheels for newer versions don't exist for Intel.

---

## 5. Configuration & Keys

### 5.1 Lookup precedence

Process environment > `.env` file (loaded by `pydantic-settings`) > built-in defaults.

### 5.2 Required keys

| Env var | Required for | Notes |
|---------|--------------|-------|
| `ANTHROPIC_API_KEY` | `STEUER_RAG_LLM_PROVIDER=anthropic` (default) | https://console.anthropic.com/ |
| `OPENAI_API_KEY` | `STEUER_RAG_LLM_PROVIDER=openai` | https://platform.openai.com/ |

### 5.3 Optional tunables (with defaults)

See `.env.example` — everything `STEUER_RAG_*` is optional and ships with sensible defaults.

| Var | Default | Notes |
|-----|---------|-------|
| `STEUER_RAG_EMBED_DEVICE` | `cpu` | Set `cuda` or `mps` for GPU |
| `STEUER_RAG_REQUEST_DELAY_MS` | `300` | Raise to `1000` if getting rate-limited |
| `STEUER_RAG_MAX_CONCURRENCY` | `5` | Parallel requests per scraper |
| `STEUER_RAG_USER_AGENT` | generic | **Set to a contactable email** before crawling |
| `STEUER_RAG_RERANK_ENABLED` | `true` | Disable for faster dev iteration |

---

## 6. Dependencies & Infrastructure

### 6.1 Python / OS

- **Python:** `>=3.11, <3.13`
- **Package manager:** `pip` or `uv` — both work; `uv` recommended for reproducible installs
- **OS:** Linux, macOS, Windows (with Python ≥ 3.11)

### 6.2 Core libraries

| Library | Role |
|---------|------|
| `langchain`, `langchain-core`, `langchain-community` | Orchestration |
| `langchain-chroma` | Vector store integration |
| `langchain-huggingface` | Embedding loader |
| `langchain-anthropic` / `langchain-openai` | LLM clients |
| `chromadb` | Persistent vector + metadata store |
| `sentence-transformers`, `torch` | Embedding + cross-encoder runtime |
| `rank_bm25` | Sparse retrieval |
| `httpx`, `beautifulsoup4`, `lxml`, `trafilatura` | Async fetch + HTML cleanup |
| `pdfplumber`, `pypdf` | PDF parsing |
| `langdetect` | Language detection |
| `pydantic`, `pydantic-settings` | Typed config + schemas |
| `typer`, `rich` | CLI |
| `fastapi`, `uvicorn` | HTTP server |
| `tenacity` | Network retries |

### 6.3 External services

| Service | Purpose | Auth |
|---------|---------|------|
| Anthropic API | Generation (default) | `ANTHROPIC_API_KEY` |
| OpenAI API | Generation (alt) | `OPENAI_API_KEY` |
| BMF, Elster, BZSt, gesetze-im-internet.de | Source content | none — public portals |
| Hugging Face Hub | First-time model download | none (anonymous) |

---

## 7. Reproducibility Guide (from scratch)

### 7.1 Prerequisites

```bash
# macOS
brew install python@3.12 uv

# Linux (Debian/Ubuntu)
sudo apt-get install python3.12 python3.12-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 7.2 Clone + install

```bash
git clone <repo-url> steuer-rsb
cd steuer-rsb

# with uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# or plain pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 7.3 Configure

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (or OPENAI_API_KEY) and a contact email
# in STEUER_RAG_USER_AGENT.
```

### 7.4 Ingest the corpus

```bash
# safe smoke test — 5 pages per source
steuer-rsb ingest all --limit 5

# full ingest (30–60 min depending on network + CPU; models download on first run)
steuer-rsb ingest all

# or ingest one source at a time
steuer-rsb ingest bundesfinanzministerium
steuer-rsb ingest gesetze
```

The first run downloads:

- BGE-M3 (~2.3 GB) into the HF cache
- BGE-reranker-v2-m3 (~600 MB) into the HF cache

Subsequent runs are local-only for the models. Previously-fetched pages are served from
`./data/raw/<source>/` — only new/changed URLs hit the network.

### 7.5 Verify retrieval

```bash
steuer-rsb info

# Deutsch
steuer-rsb search "Werbungskosten Homeoffice Pauschale" --k 5
steuer-rsb ask "Wann muss ich die Steuererklärung abgeben?"

# English
steuer-rsb ask "How do I file an income tax return in Germany as an employee?"

# Source filter
steuer-rsb search "§ 33a EStG Unterhalt Ausland" --source gesetze --k 5
steuer-rsb search "Lohnsteuer" --source bzst --k 5
```

### 7.6 Run the API server

```bash
steuer-rsb serve --host 0.0.0.0 --port 8000

# in another terminal
curl -X POST http://localhost:8000/ask \
  -H "content-type: application/json" \
  -d '{"question": "Welche Belege muss ich aufbewahren?", "k": 6}'
```

OpenAPI docs at <http://localhost:8000/docs>.

### 7.7 Minimum reproducible RAG (one paste)

```bash
git clone <repo-url> steuer-rsb && cd steuer-rsb
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env && $EDITOR .env       # set ANTHROPIC_API_KEY
steuer-rsb ingest all --limit 5
steuer-rsb ask "Bis wann muss ich die Steuererklärung 2024 abgeben?"
```

If the last step returns a cited answer, the system is operational.

---

## 8. Operational Runbook

### 8.1 Empty results / "context does not answer"

Likely causes:

1. Ingest hasn't run yet → `steuer-rag info` shows `indexed_chunks: 0`.
2. The query is in a language under-represented in the corpus — pass `--lang de` or `--lang en`
   to skip the auto-detect.
3. The query is about a very specific statutory provision — try `--source gesetze` to search
   the EStG / AO law text directly.

### 8.2 Scraper getting blocked / rate-limited

Bump politeness: `STEUER_RAG_REQUEST_DELAY_MS=1000` and `STEUER_RAG_MAX_CONCURRENCY=2`. Confirm
your `STEUER_RAG_USER_AGENT` includes a working contact email — ministerial portals tolerate
identifiable bots far better than anonymous ones.

### 8.3 Stale corpus

Re-run `steuer-rag ingest all` on a weekly cron. Chunk IDs are content-addressable so unchanged
content is a no-op on Chroma; only deltas re-index.

### 8.4 Re-embedding (model swap)

```bash
rm -rf data/chroma
# edit .env: STEUER_RAG_EMBED_MODEL=intfloat/multilingual-e5-large
steuer-rsb ingest all
```

### 8.5 Tests

```bash
pytest -q
```

The included tests cover chunking determinism, doc-ID stability, and language detection.

---

## 9. Appendix — Source Catalog

| Source ID | File | Tags | Languages | Scope |
|-----------|------|------|-----------|-------|
| `bundesfinanzministerium` | `sources/bmf.py` | BMF-Schreiben, brochures, FAQ | DE, EN | Einkommensteuer, Lohnsteuer, Kapitalertragsteuer, Erbschaft — individual filers only; excludes VAT, corporate, fiscal statistics |
| `elster` | `sources/elster.py` | filing portal, help | DE | Public `eportal/infoseite`, `hilfe`, `formulare-leistungen` |
| `bzst` | `sources/bzst.py` | tax authority, forms, PDF | DE, EN | `Privatpersonen` — IdNr, ELStAM, Altersvorsorge, Kapitalerträge, Kirchensteuer |
| `gesetze` | `sources/gesetze.py` | law text | DE | EStG (§§ 1–139), AO (§§ 1–414), EStDV — section-level pages from gesetze-im-internet.de |

All four sources feed the shared `steuer_chunks` Chroma collection, so a single retrieval call
returns hits across every ingested source unless `--source` is set.

---

### Document control

- **Last reviewed:** 2026-05-14
- **Owner:** Steuer-RAG maintainers
- **Source of truth:** `steuer_rag/sources/registry.py`, `steuer_rag/config/settings.py`,
  `steuer_rag/schema/models.py`, `pyproject.toml`. Any conflict between this doc and those
  files: **the code wins**, then file a doc PR.

### Disclaimer

This system is an information-retrieval tool over publicly-available, official German tax
material. It is **not legal or tax advice**. For binding guidance on individual cases,
consult a `Steuerberater` (tax advisor) or your local `Finanzamt`.