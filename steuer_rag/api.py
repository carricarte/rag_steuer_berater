"""FastAPI server exposing `/ask`, `/search`, and `/health`.

Production niceties: structured logging, request validation, CORS, streaming optional via SSE.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from steuer_rag.config import get_settings
from steuer_rag.generation.chain import ask as run_ask
from steuer_rag.pipeline.index import get_index
from steuer_rag.retrieval.search import search as run_search

log = logging.getLogger(__name__)

app = FastAPI(
    title="Steuer-RAG",
    version="0.1.0",
    description="Bilingual (DE/EN) RAG for the German Steuererklärung.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- request/response models -----


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(default=8, ge=1, le=25)
    source: str | None = None
    language: str | None = None


class CitationOut(BaseModel):
    n: int
    source: str
    url: str
    title: str
    section: str | None = None
    rerank_score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    language: str
    citations: list[CitationOut]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    k: int = Field(default=8, ge=1, le=50)
    strategy: str = "hybrid_rerank"
    source: str | None = None
    language: str | None = None


class SearchHitOut(BaseModel):
    score: float | None = None
    source: str
    language: str
    url: str
    title: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    strategy: str
    query_language: str
    hits: list[SearchHitOut]


# ----- endpoints -----


@app.get("/health")
def health() -> dict:
    s = get_settings()
    api_key_set = bool(
        s.anthropic_api_key if s.llm_provider == "anthropic" else s.openai_api_key
    )
    return {
        "status": "ok",
        "indexed_chunks": get_index().count(),
        "embed_model": s.embed_model,
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "api_key_set": api_key_set,
        "snapshot_dataset": s.snapshot_dataset,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = run_ask(req.question, k=req.k, source=req.source, language=req.language)
    except Exception as e:
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    return AskResponse(
        question=result.question,
        answer=result.answer,
        language=result.language.value,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    result = run_search(
        req.query,
        k=req.k,
        strategy=req.strategy,
        source=req.source,
        language=req.language,
    )
    hits: list[SearchHitOut] = []
    for d in result.documents:
        m = d.metadata or {}
        hits.append(
            SearchHitOut(
                score=m.get("rerank_score"),
                source=m.get("source", ""),
                language=m.get("language", ""),
                url=m.get("url", ""),
                title=m.get("doc_title", ""),
                snippet=d.page_content[:400],
            )
        )
    return SearchResponse(
        query=req.query,
        strategy=result.strategy,
        query_language=result.query_language.value,
        hits=hits,
    )
