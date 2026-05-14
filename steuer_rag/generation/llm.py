"""LLM factory. Switches Anthropic vs OpenAI based on settings."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from steuer_rag.config import get_settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    s = get_settings()
    if s.llm_provider == "anthropic":
        if not s.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in .env or switch STEUER_RAG_LLM_PROVIDER."
            )
        from langchain_anthropic import ChatAnthropic

        log.info("LLM: anthropic / %s", s.llm_model)
        kwargs: dict = {
            "model": s.llm_model,
            "max_tokens": s.llm_max_tokens,
            "api_key": s.anthropic_api_key.get_secret_value(),
        }
        if s.llm_temperature is not None:
            kwargs["temperature"] = s.llm_temperature
        return ChatAnthropic(**kwargs)

    if s.llm_provider == "openai":
        if not s.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in .env or switch STEUER_RAG_LLM_PROVIDER."
            )
        from langchain_openai import ChatOpenAI

        log.info("LLM: openai / %s", s.llm_model)
        kwargs = {
            "model": s.llm_model,
            "max_tokens": s.llm_max_tokens,
            "api_key": s.openai_api_key.get_secret_value(),
        }
        if s.llm_temperature is not None:
            kwargs["temperature"] = s.llm_temperature
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unknown llm_provider: {s.llm_provider}")
