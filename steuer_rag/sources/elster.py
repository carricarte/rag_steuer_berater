"""Scraper for elster.de — the official online tax-filing portal (Mein ELSTER).

ELSTER's content is largely behind login, but public help pages, FAQs, glossary, and download
landing pages cover the most-asked questions about preparing a Steuererklärung. We stick to those
public areas.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from steuer_rag.schema.models import DocumentCore, SourceName
from steuer_rag.sources.base import BaseScraper

log = logging.getLogger(__name__)


class ElsterScraper(BaseScraper):
    source = SourceName.ELSTER
    base_url = "https://www.elster.de"

    # Verified 2026-05-14. Most ELSTER content is auth-walled; only a few public entry-points
    # exist. We rely on link discovery from these landing pages.
    # Note: ELSTER renders much of its UI client-side, so a non-JS scraper will see less than a
    # browser would. The seeds below are server-rendered.
    seed_paths = (
        "/eportal/start",
        "/eportal/registrierung-auswahl",
        "/eportal/formulare-leistungen",
    )

    allow_pattern = re.compile(
        r"elster\.de/eportal/("
        r"infoseite|hilfe|wo_finde_ich|formulare-leistungen|registrierung"
        r"|allgemeines|datenschutz|service"
        r").*",
        re.I,
    )

    max_pages = 200

    async def crawl(self) -> AsyncIterator[DocumentCore]:
        to_visit: list[str] = []
        for path in self.seed_paths:
            url = self.base_url + path
            resp = await self.fetch(url)
            if resp is None:
                continue
            html = resp.content.decode("utf-8", errors="ignore")
            to_visit.extend(self.discover_links(html, base=url))
            doc = await self.fetch_and_parse(url, section="seed")
            if doc:
                yield doc

        deduped = list(dict.fromkeys(to_visit))
        log.info("[elster] discovered %d candidate links", len(deduped))

        for url in deduped[: self.max_pages]:
            doc = await self.fetch_and_parse(url)
            if doc:
                yield doc
