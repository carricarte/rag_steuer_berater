"""Scraper for elster.de — the official online tax-filing portal (Mein ELSTER).

ELSTER's content is largely behind login, but public help pages, FAQs, glossary, and download
landing pages cover the most-asked questions about preparing a Steuererklärung. We stick to those
public areas. Inherits depth-N BFS + PDF harvesting from `BaseScraper`.
"""

from __future__ import annotations

import logging
import re

from steuer_rag.schema.models import SourceName
from steuer_rag.sources.base import BaseScraper

log = logging.getLogger(__name__)


class ElsterScraper(BaseScraper):
    source = SourceName.ELSTER
    base_url = "https://www.elster.de"

    # Verified 2026-05-14. Most ELSTER content is auth-walled; only a few public entry-points
    # exist. ELSTER renders much UI client-side, so a non-JS scraper sees less than a browser.
    # The seeds below are server-rendered.
    seed_paths = (
        "/eportal/start",
        "/eportal/registrierung-auswahl",
        "/eportal/formulare-leistungen",
    )

    allow_pattern = re.compile(
        r"elster\.de/eportal/("
        r"infoseite|hilfe|wo_finde_ich|formulare-leistungen|registrierung"
        r"|allgemeines|datenschutz|service|start"
        r").*",
        re.I,
    )

    # ELSTER does have downloadable PDFs (e.g., user guides) under /eportal/.
    pdf_allow_pattern = re.compile(r"elster\.de/.*\.pdf", re.I)

    max_pages = 250
    max_depth = 2  # ELSTER public surface is small; depth 2 is plenty.
