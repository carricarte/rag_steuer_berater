"""Steuer-RAG — bilingual (DE/EN) retrieval-augmented system for German Steuererklärung.

Public surface:
- `settings`         — runtime configuration (env-driven)
- `pipeline.ingest`  — end-to-end ingest from official sources
- `retrieval.search` — hybrid retriever entry point
- `generation.chain` — RAG answer chain
"""

__version__ = "0.1.0"