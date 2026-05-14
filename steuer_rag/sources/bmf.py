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

    # HTML scope: individual-taxpayer tax content only. Negative lookahead excludes:
    # - VAT / corporate / trade tax (Umsatzsteuer, Gewerbesteuer, Koerperschaft)
    # - fiscal statistics (Steuerschaetzung, Steuereinnahmen, Umrechnungskurse)
    # - non-content pages (Pressemitteilungen, Bilderstrecken, Bilder/, Video-Textfassungen,
    #   Glossareintraege, Standardartikel/Meta)
    # - business-only guidance (AfA-Tabelle, Betriebspruefung)
    # - pagination (VollstaendigeListe)
    allow_pattern = re.compile(
        r"bundesfinanzministerium\.de/(Content|Web)/(DE|EN)/"
        r"(?!.*(Umsatzsteuer|Gewerbesteuer|Koerperschaft|Betriebspruefung|AfA-Tabelle|"
        r"Steuerschaetzung|Steuereinnahmen|Pressemitteilung|Bilderstrecken|"
        r"Bilder/|Video-Textfassung|Glossareintraeg|Standardartikel/Meta|"
        r"VollstaendigeListe|Umrechnungskurse))"
        r".*"
        r"(Steuer|Einkommen|Lohn|Abgabe|Tax|Income|Wage|Levy|Erklaerung|Declaration|FAQ)",
        re.I,
    )

    # PDF scope: BMF-Schreiben (binding administrative guidance), Broschueren, FAQ bundles,
    # and the EN Press_Room/Publications brochures. `Standardartikel` is intentionally excluded —
    # it covers the monthly Steuereinnahmen and Steuerschätzung fiscal-statistics PDFs which
    # are irrelevant to individual tax filing.
    pdf_allow_pattern = re.compile(
        r"bundesfinanzministerium\.de/Content/(DE|EN)/Downloads/.*"
        r"(BMF_Schreiben|Broschueren|FAQ).*\.pdf"
        r"|bundesfinanzministerium\.de/Content/(DE|EN)/Standardartikel/Press_Room/Publications/.*\.pdf",
        re.I,
    )

    max_pages = 500
    max_depth = 3
