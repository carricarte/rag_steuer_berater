"""Registry: maps source name → scraper factory. Same pattern as the reference design."""

from __future__ import annotations

from collections.abc import Callable

from steuer_rag.schema.models import SourceName
from steuer_rag.sources.base import BaseScraper
from steuer_rag.sources.bundesfinanzministerium import BMFScraper
from steuer_rag.sources.bzst import BZStScraper
from steuer_rag.sources.elster import ElsterScraper

SourceFactory = Callable[[], BaseScraper]

REGISTRY: dict[SourceName, SourceFactory] = {
    SourceName.BMF: BMFScraper,
    SourceName.ELSTER: ElsterScraper,
    SourceName.BZST: BZStScraper,
}


def get_source(name: str | SourceName) -> BaseScraper:
    if isinstance(name, str):
        name = SourceName(name)
    if name not in REGISTRY:
        raise KeyError(f"unknown source: {name}. Valid: {[s.value for s in REGISTRY]}")
    return REGISTRY[name]()
