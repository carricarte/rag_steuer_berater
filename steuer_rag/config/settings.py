"""Env-driven runtime configuration.

All settings are loaded from environment variables (with `.env` fallback) prefixed `STEUER_RAG_`
plus the standard LLM provider keys. Use `get_settings()` everywhere — it caches a single instance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import os

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="STEUER_RAG_",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM (read both prefixed + raw keys) ----
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "STEUER_RAG_ANTHROPIC_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "STEUER_RAG_OPENAI_API_KEY"),
    )
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-opus-4-7"
    # None = let the model use its default. Newer reasoning models (opus-4-7, sonnet-4-6 …)
    # reject `temperature`, so we only send it when explicitly set.
    llm_temperature: float | None = None
    llm_max_tokens: int = 1024

    # ---- Embedding ----
    embed_model: str = "BAAI/bge-m3"
    embed_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embed_normalize: bool = True

    # ---- Reranker ----
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 50
    rerank_enabled: bool = True

    # ---- Vector store ----
    vector_backend: Literal["chroma"] = "chroma"
    chroma_dir: Path = Path("./data/chroma")
    collection: str = "steuer_chunks"

    # ---- Scraper ----
    user_agent: str = "steuer-rsb/0.1 (+contact: you@example.com)"
    request_delay_ms: int = 400
    max_concurrency: int = 4
    raw_dir: Path = Path("./data/raw")
    scraper_timeout_s: float = 30.0
    scraper_max_retries: int = 3

    # ---- Chunking ----
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_chunk_chars: int = 200
    chunk_strategy_version: str = "v1"

    # ---- Retrieval ----
    top_k: int = 8
    hybrid_dense_weight: float = 0.6
    hybrid_sparse_weight: float = 0.4
    candidate_multiplier: int = 4  # over-fetch factor before rerank/filter

    # ---- Snapshot persistence ----
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "STEUER_RAG_HF_TOKEN"),
    )
    snapshot_dataset: str | None = Field(
        default=None,
        validation_alias=AliasChoices("STEUER_RAG_SNAPSHOT_DATASET", "SNAPSHOT_DATASET"),
    )

    # ---- Logging ----
    log_level: str = "INFO"

    # AliasChoices can have subtle interactions with env_prefix; this validator
    # reads the standard provider keys directly from os.environ as a fallback.
    @model_validator(mode="after")
    def _fallback_api_keys(self) -> "Settings":
        if self.anthropic_api_key is None and (v := os.environ.get("ANTHROPIC_API_KEY")):
            object.__setattr__(self, "anthropic_api_key", SecretStr(v))
        if self.openai_api_key is None and (v := os.environ.get("OPENAI_API_KEY")):
            object.__setattr__(self, "openai_api_key", SecretStr(v))
        if self.hf_token is None and (v := os.environ.get("HF_TOKEN")):
            object.__setattr__(self, "hf_token", SecretStr(v))
        return self

    # --- derived helpers ---
    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings (cached)."""
    s = Settings()
    s.ensure_dirs()
    return s
