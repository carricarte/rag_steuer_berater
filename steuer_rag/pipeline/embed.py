"""Embedding model loader. Multilingual by default (BGE-M3, 1024-d).

Wraps HuggingFace sentence-transformers via `langchain_huggingface.HuggingFaceEmbeddings` so it
plugs directly into LangChain retrievers and vectorstores. Loaded once, cached per-process.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from steuer_rag.config import get_settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    s = get_settings()
    log.info("loading embedding model %s on %s", s.embed_model, s.embed_device)
    return HuggingFaceEmbeddings(
        model_name=s.embed_model,
        model_kwargs={"device": s.embed_device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": s.embed_normalize, "batch_size": 32},
    )
