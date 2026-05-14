"""Hybrid retrieval: dense (Chroma) + sparse (BM25) → RRF fusion via LangChain EnsembleRetriever
→ cross-encoder rerank. Language-aware filtering: prefer same-language passages when caller asks
for one, but fall back to the other language if the index is thin.

Mirrors the four-strategy layout in the reference design (fts, vector, hybrid, hybrid_rerank)
but exposes a single `search()` entry point with `strategy=` because LangChain composes the
underlying retrievers for us.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

try:
    # langchain >= 1.0 moved EnsembleRetriever to langchain_classic
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover — older langchain (<1.0)
    from langchain.retrievers import EnsembleRetriever  # type: ignore[no-redef]
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from steuer_rag.config import get_settings
from steuer_rag.pipeline.index import get_index
from steuer_rag.retrieval.reranker import rerank
from steuer_rag.schema.models import Language, SourceName, detect_language

log = logging.getLogger(__name__)

Strategy = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]


@dataclass(slots=True)
class RetrievalResult:
    documents: list[Document]
    strategy: Strategy
    query_language: Language


# -----------------------------------------------------------------------------
# BM25 — built in-process from the indexed chunks. Cached and invalidated by
# the total document count so newly-ingested chunks are picked up.
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _bm25_for_count(count: int) -> BM25Retriever:
    docs = get_index().all_documents()
    log.info("[bm25] building over %d documents", len(docs))
    retriever = BM25Retriever.from_documents(docs or [Document(page_content="")])
    return retriever


def _bm25_retriever(k: int) -> BM25Retriever:
    idx = get_index()
    r = _bm25_for_count(idx.count())
    r.k = k
    return r


# -----------------------------------------------------------------------------


class HybridRetriever:
    """Composable retriever: dense + sparse + optional rerank."""

    def __init__(
        self,
        *,
        k: int = 8,
        strategy: Strategy = "hybrid_rerank",
        source_filter: SourceName | None = None,
        language_preference: Language | None = None,
    ) -> None:
        self.settings = get_settings()
        self.k = k
        self.strategy = strategy
        self.source_filter = source_filter
        self.language_preference = language_preference

    # ----- internal helpers -----

    def _chroma_filter(self) -> dict | None:
        clauses: list[dict] = []
        if self.source_filter:
            clauses.append({"source": self.source_filter.value})
        # NOTE: we don't hard-filter on language here — we prefer it via rerank/sort below so the
        # caller still gets results when the requested language is sparsely indexed.
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _dense(self, query: str, *, k: int) -> list[Document]:
        retriever = get_index().as_retriever(k=k, filter=self._chroma_filter())
        return retriever.invoke(query)

    def _sparse(self, query: str, *, k: int) -> list[Document]:
        retriever = _bm25_retriever(k * (3 if self.source_filter else 1))
        results = retriever.invoke(query)
        if self.source_filter:
            results = [
                d for d in results if d.metadata.get("source") == self.source_filter.value
            ]
        return results[:k]

    def _hybrid(self, query: str, *, k: int) -> list[Document]:
        dense_retriever = get_index().as_retriever(k=k, filter=self._chroma_filter())
        sparse = _bm25_retriever(k * (3 if self.source_filter else 1))
        ens = EnsembleRetriever(
            retrievers=[dense_retriever, sparse],
            weights=[self.settings.hybrid_dense_weight, self.settings.hybrid_sparse_weight],
        )
        docs = ens.invoke(query)
        if self.source_filter:
            docs = [d for d in docs if d.metadata.get("source") == self.source_filter.value]
        return docs[:k]

    def _prefer_language(self, docs: list[Document]) -> list[Document]:
        if not self.language_preference or self.language_preference == Language.UNKNOWN:
            return docs
        target = self.language_preference.value
        preferred = [d for d in docs if d.metadata.get("language") == target]
        other = [d for d in docs if d.metadata.get("language") != target]
        return preferred + other

    # ----- public -----

    def invoke(self, query: str) -> RetrievalResult:
        ql = detect_language(query)
        # If caller hasn't set a preference, infer it from the query language.
        if self.language_preference is None:
            self.language_preference = ql

        if self.strategy == "dense":
            docs = self._dense(query, k=self.k * self.settings.candidate_multiplier)
        elif self.strategy == "sparse":
            docs = self._sparse(query, k=self.k * self.settings.candidate_multiplier)
        else:
            # hybrid / hybrid_rerank
            docs = self._hybrid(query, k=self.settings.rerank_top_n)

        docs = self._prefer_language(docs)

        if self.strategy == "hybrid_rerank":
            docs = rerank(query, docs, k=self.k)
        else:
            docs = docs[: self.k]

        return RetrievalResult(documents=docs, strategy=self.strategy, query_language=ql)


def build_retriever(
    *,
    k: int | None = None,
    strategy: Strategy = "hybrid_rerank",
    source: str | SourceName | None = None,
    language: str | Language | None = None,
) -> HybridRetriever:
    s = get_settings()
    src = SourceName(source) if isinstance(source, str) else source
    lang = Language(language) if isinstance(language, str) else language
    return HybridRetriever(
        k=k or s.top_k,
        strategy=strategy,
        source_filter=src,
        language_preference=lang,
    )


def search(query: str, **kwargs) -> RetrievalResult:
    return build_retriever(**kwargs).invoke(query)
