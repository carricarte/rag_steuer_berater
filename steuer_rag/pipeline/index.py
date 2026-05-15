"""Vector store wrapper. Chroma is the default backend (persistent, embedded, no external service).

Exposes `add_chunks` (idempotent — keyed by chunk_id) and a typed `as_retriever()` helper. All
metadata lands as flat strings/ints so Chroma's filtering works.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from steuer_rag.config import get_settings
from steuer_rag.pipeline.embed import get_embeddings
from steuer_rag.schema.models import DocumentChunk

log = logging.getLogger(__name__)


class VectorIndex:
    """Thin wrapper around a LangChain `VectorStore` with chunk-aware helpers."""

    # Keep well under Chroma's hard cap (~5461) and limit peak memory: each batch is
    # embedded + written to disk before the next is loaded, so we hold at most 500
    # chunk texts + their float32 vectors in RAM at any time.
    _WRITE_BATCH_SIZE: int = 500

    def __init__(self, store: VectorStore) -> None:
        self.store = store

    # ----- write path -----

    def add_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        import gc

        written = 0
        batch_docs: list[Document] = []
        batch_ids: list[str] = []

        for c in chunks:
            batch_docs.append(Document(page_content=c.content, metadata=c.to_metadata()))
            batch_ids.append(c.chunk_id)

            if len(batch_docs) >= self._WRITE_BATCH_SIZE:
                self.store.add_documents(documents=batch_docs, ids=batch_ids)
                written += len(batch_docs)
                log.info("[index] wrote %d chunks (running total %d)", len(batch_docs), written)
                batch_docs = []
                batch_ids = []
                gc.collect()

        if batch_docs:
            self.store.add_documents(documents=batch_docs, ids=batch_ids)
            written += len(batch_docs)
            log.info("[index] wrote %d chunks (running total %d)", len(batch_docs), written)
            gc.collect()

        return written

    # ----- read path -----

    def as_retriever(self, *, k: int = 8, filter: dict | None = None):
        kwargs = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return self.store.as_retriever(search_kwargs=kwargs)

    def count(self) -> int:
        """Best-effort row count (Chroma)."""
        try:
            return self.store._collection.count()  # type: ignore[attr-defined]
        except Exception:
            return -1

    def all_documents(self) -> list[Document]:
        """Pull every document. Used to build BM25 in-process — fine for our scale (~10k chunks)."""
        try:
            col = self.store._collection  # type: ignore[attr-defined]
            data = col.get(include=["documents", "metadatas"])
            docs: list[Document] = []
            for content, meta in zip(data.get("documents", []), data.get("metadatas", [])):
                docs.append(Document(page_content=content, metadata=meta or {}))
            return docs
        except Exception as e:
            log.warning("all_documents fallback (%s)", e)
            return []


@lru_cache(maxsize=1)
def get_index() -> VectorIndex:
    s = get_settings()
    store = Chroma(
        collection_name=s.collection,
        embedding_function=get_embeddings(),
        persist_directory=str(s.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    return VectorIndex(store)
