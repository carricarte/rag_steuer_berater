"""Scraper for bzst.de — Bundeszentralamt für Steuern (Federal Central Tax Office).

BZSt publishes detailed guidance, downloadable forms (PDFs), and FAQs in both German and English
under `/DE/` and `/EN/` mirrors. We harvest both.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from steuer_rag.schema.models import DocumentCore, SourceName
from steuer_rag.sources.base import BaseScraper

log = logging.getLogger(__name__)


class BZStScraper(BaseScraper):
    source = SourceName.BZST
    base_url = "https://www.bzst.de"

    # Verified 2026-05-14. BZSt reorganized several Privatpersonen subtopics; the index page
    # links to all current subtopics, so link discovery covers them.
    seed_paths = (
        # German
        "/DE/Privatpersonen/privatpersonen_node.html",
        "/DE/Privatpersonen/SteuerlicheIdentifikationsnummer/steuerlicheidentifikationsnummer_node.html",
        "/DE/Service/Behoerdenwegweiser/behoerdenwegweiser_node.html",
        "/DE/Service/service_node.html",
        # English
        "/EN/Home/home_node.html",
        "/EN/Service/service_node.html",
    )

    allow_pattern = re.compile(
        r"bzst\.de/(DE|EN)/.*"
        r"(Steuer|Einkommen|Lohn|Tax|Income|Wage|Erklaerung|Declaration|FAQ|Service|Privatpersonen|PrivateIndividuals)",
        re.I,
    )

    max_pages = 250

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
        log.info("[bzst] discovered %d candidate links", len(deduped))

        for url in deduped[: self.max_pages]:
            doc = await self.fetch_and_parse(url)
            if doc:
                yield doc
