"""End-to-end ingest: source crawl → chunk → embed → write to index.

Run this once per source (or all sources) to (re-)populate the retrieval index. Re-runs are
idempotent because doc_id and chunk_id are content-addressable.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from steuer_rag.config import get_settings
from steuer_rag.pipeline.index import VectorIndex, get_index
from steuer_rag.schema.chunking import chunks_for_document
from steuer_rag.schema.models import DocumentCore, SourceName
from steuer_rag.sources.registry import REGISTRY, get_source

log = logging.getLogger(__name__)


async def _drain_source(source: SourceName, *, limit: int | None = None) -> list[DocumentCore]:
    docs: list[DocumentCore] = []
    async with get_source(source) as scraper:
        async for doc in scraper.crawl():
            docs.append(doc)
            if limit and len(docs) >= limit:
                break
    return docs


def _chunk_docs(docs: Iterable[DocumentCore]):
    s = get_settings()
    total = 0
    for d in docs:
        for chunk in chunks_for_document(
            d,
            chunk_size=s.chunk_size,
            overlap=s.chunk_overlap,
            min_chunk_chars=s.min_chunk_chars,
            strategy_version=s.chunk_strategy_version,
        ):
            total += 1
            yield chunk
    log.info("[chunker] produced %d chunks", total)


async def ingest_source(
    source: SourceName | str,
    *,
    limit: int | None = None,
    index: VectorIndex | None = None,
) -> dict:
    if isinstance(source, str):
        source = SourceName(source)
    index = index or get_index()
    log.info("[ingest] starting %s (limit=%s)", source.value, limit)
    docs = await _drain_source(source, limit=limit)
    log.info("[ingest] %s — %d docs fetched", source.value, len(docs))
    chunks = list(_chunk_docs(docs))
    written = index.add_chunks(chunks)
    log.info("[ingest] %s — %d chunks indexed", source.value, written)
    return {"source": source.value, "docs": len(docs), "chunks": written}


async def ingest_all(*, limit: int | None = None) -> list[dict]:
    """Sequentially ingest every registered source. Sequential to avoid embedding contention."""
    results: list[dict] = []
    index = get_index()
    for source in REGISTRY.keys():
        try:
            res = await ingest_source(source, limit=limit, index=index)
            results.append(res)
        except Exception as e:
            log.exception("[ingest] %s failed: %s", source.value, e)
            results.append({"source": source.value, "error": str(e)})
    return results


def ingest_source_sync(source: SourceName | str, **kwargs) -> dict:
    return asyncio.run(ingest_source(source, **kwargs))


def ingest_all_sync(**kwargs) -> list[dict]:
    return asyncio.run(ingest_all(**kwargs))
