"""Scraper for gesetze-im-internet.de — official BMJV publication of German federal statutes.

Indexes the three tax-relevant laws:
  - EStG (Einkommensteuergesetz)     — /estg/
  - AO  (Abgabenordnung)             — /ao_1977/
  - EStDV (EStG-Durchführungsverordnung) — /estdv/

Each law's TOC page (depth 0) links to individual section pages (depth 1), so max_depth=1 is
sufficient. Thin-page threshold is lowered to 200 chars — short legal provisions are still
indexing-worthy. Inherits BFS + PDF harvesting from BaseScraper.
"""

from __future__ import annotations

import re

from steuer_rag.schema.models import SourceName
from steuer_rag.sources.base import BaseScraper


class GesetzeImInternetScraper(BaseScraper):
    source = SourceName.GESETZE
    base_url = "https://www.gesetze-im-internet.de"

    # Verified 2026-05-14. Each TOC page links to all section pages for that law.
    seed_paths = (
        "/estg/",
        "/ao_1977/",
        "/estdv/",
    )

    # HTML scope: section pages directly under each law directory only.
    # Matches /estg/__1.html, /estg/anlage_xyz.html, /ao_1977/__1.html, etc.
    # The trailing `(/[^/]*)?$` prevents descending into subdirectories.
    allow_pattern = re.compile(
        r"gesetze-im-internet\.de/(estg|ao_1977|estdv)(/[^/]*)?$",
        re.I,
    )

    # PDF scope: consolidated law PDFs only (e.g. EStG.pdf, AO.pdf, EStDV.pdf).
    # These supplement the HTML sections for content that PDF extraction captures better
    # (e.g. tables in EStDV annexes).
    pdf_allow_pattern = re.compile(
        r"gesetze-im-internet\.de/(estg|ao_1977|estdv)/\w+\.pdf",
        re.I,
    )

    max_pages = 700  # EStG ~107 + AO ~414 + EStDV ~80 sections, plus 3 PDFs
    max_depth = 1
    thin_html_chars = 200  # short provisions (e.g. § 2 AO, one sentence) are still valuable