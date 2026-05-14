"""Scraper for bundesfinanzministerium.de — BMF official portal.

Focuses on the German tax declaration (Steuererklärung) content area plus its English mirror under
`/Web/EN/`. Inherits the depth-N BFS + PDF harvesting from `BaseScraper`.
"""

from __future__ import annotations

import logging
import re

from steuer_rag.schema.models import SourceName
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

    # HTML scope: only tax-related topic + content paths in DE or EN.
    allow_pattern = re.compile(
        r"bundesfinanzministerium\.de/(Content|Web)/(DE|EN)/.*"
        r"(Steuer|Einkommen|Lohn|Abgabe|Tax|Income|Wage|Levy|Erklaerung|Declaration|FAQ)",
        re.I,
    )

    # PDF scope: BMF-Schreiben (binding administrative guidance), publication brochures
    # (Broschueren_Bestellservice / Press_Room/Publications), and FAQ download bundles.
    # Everything else (press releases, image stories, raw forms) is intentionally excluded.
    pdf_allow_pattern = re.compile(
        r"bundesfinanzministerium\.de/Content/(DE|EN)/.*"
        r"(BMF_Schreiben|Broschueren|Brochures|Publications|FAQ|Standardartikel).*\.pdf",
        re.I,
    )

    max_pages = 500
    max_depth = 3
