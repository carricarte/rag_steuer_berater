"""Scraper for bundesfinanzministerium.de — BMF official portal.

Focuses on the German tax declaration (Steuererklärung) content area plus its English mirror under
`/Web/EN/`. Discovery is two-step: seed pages → links within scope → fetch+parse.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from steuer_rag.schema.models import DocumentCore, SourceName
from steuer_rag.sources.base import BaseScraper

log = logging.getLogger(__name__)


class BMFScraper(BaseScraper):
    source = SourceName.BMF
    base_url = "https://www.bundesfinanzministerium.de"

    # Verified 2026-05-14. BMF restructured several FAQ paths — link discovery from these
    # topic landing pages reaches equivalent content. 503s on FAQ endpoints are bot-protection
    # blocks; we intentionally don't list those as seeds.
    seed_paths = (
        # German topic area — Steuern + Steuerarten + key Steuerart pages
        "/Web/DE/Themen/Steuern/steuern.html",
        "/Web/DE/Themen/Steuern/Steuerarten/steuerarten.html",
        "/Web/DE/Themen/Steuern/Steuerarten/Einkommensteuer/einkommensteuer.html",
        "/Web/DE/Themen/Steuern/Steuerarten/Lohnsteuer/lohnsteuer.html",
        # English mirror
        "/Web/EN/Issues/Taxation/taxation.html",
    )

    allow_pattern = re.compile(
        r"bundesfinanzministerium\.de/(Content|Web)/(DE|EN)/.*"
        r"(Steuer|Einkommen|Lohn|Abgabe|Tax|Income|Wage|Levy|Erklaerung|Declaration|FAQ)",
        re.I,
    )

    max_pages = 250

    async def crawl(self) -> AsyncIterator[DocumentCore]:
        # 1) fetch seeds and extract in-scope links
        to_visit: list[str] = []
        for path in self.seed_paths:
            url = self.base_url + path
            resp = await self.fetch(url)
            if resp is None:
                continue
            html = resp.content.decode("utf-8", errors="ignore")
            to_visit.extend(self.discover_links(html, base=url))
            # yield the seed page itself
            doc = await self.fetch_and_parse(url, section="seed")
            if doc:
                yield doc

        # 2) crawl discovered links breadth-first, capped by max_pages
        # dedupe while preserving order
        deduped = list(dict.fromkeys(to_visit))
        log.info("[bmf] discovered %d candidate links", len(deduped))

        for url in deduped[: self.max_pages]:
            doc = await self.fetch_and_parse(url)
            if doc:
                yield doc
