"""Cross-encoder reranker — multilingual (BGE-reranker-v2-m3 by default).

Lazy singleton. Wraps `sentence-transformers.CrossEncoder` and exposes a tiny `rerank()` method
that returns the top-`k` documents by (query, passage) score.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document

from steuer_rag.config import get_settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder  # local import — heavy

    s = get_settings()
    log.info("loading reranker %s", s.rerank_model)
    return CrossEncoder(s.rerank_model, max_length=512, device=s.embed_device)


def rerank(query: str, docs: list[Document], *, k: int) -> list[Document]:
    """Return top-`k` docs ordered by cross-encoder relevance to `query`."""
    if not docs:
        return []
    s = get_settings()
    if not s.rerank_enabled or len(docs) <= 1:
        return docs[:k]
    ce = _load_cross_encoder()
    pairs = [(query, d.page_content) for d in docs]
    scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
    out: list[Document] = []
    for score, doc in ranked[:k]:
        # attach the rerank score so the chain can show it
        meta = dict(doc.metadata or {})
        meta["rerank_score"] = float(score)
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out
