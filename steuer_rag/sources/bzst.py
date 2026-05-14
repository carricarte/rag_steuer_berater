"""Scraper for bzst.de — Bundeszentralamt für Steuern (Federal Central Tax Office).

BZSt publishes detailed guidance, downloadable forms (PDFs), and FAQs in both German and English
under `/DE/` and `/EN/` mirrors. Inherits depth-N BFS + PDF harvesting from `BaseScraper`.
"""

from __future__ import annotations

import logging
import re

from steuer_rag.schema.models import SourceName
from steuer_rag.sources.base import BaseScraper

log = logging.getLogger(__name__)


class BZStScraper(BaseScraper):
    source = SourceName.BZST
    base_url = "https://www.bzst.de"

    # Verified 2026-05-14. Index pages link to all current subtopics, so depth-N crawl reaches them.
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

    # HTML scope: stay on the individual-taxpayer side of bzst.de.
    # - /DE/Privatpersonen/* and /EN/PrivateIndividuals/* are accepted UNCONDITIONALLY
    #   (every subtopic here — ELStAM, Kontenabruf, Kindergeld, IdNr, etc. — is in scope).
    # - /EN/Home/* is the EN landing area, accepted unconditionally.
    # - /DE/Service/* and /EN/Service/* are accepted only when the URL also contains a
    #   tax-related token, so we keep FAQ/Behoerdenwegweiser but skip pure site machinery
    #   (RecommendPage, Sitemap, Accessibility, etc.).
    # - /DE/Unternehmen/, /DE/Behoerden/, and any other top-level area are excluded.
    allow_pattern = re.compile(
        r"bzst\.de/("
        r"DE/Privatpersonen/.*"
        r"|EN/PrivateIndividuals/.*"
        r"|EN/Home/.*"
        r"|(DE|EN)/Service/.*("
        r"Steuer|Einkommen|Lohn|Tax|Income|Wage|Erklaerung|Declaration|FAQ|"
        r"Behoerden|Wegweiser|Identifikationsnummer|TaxID|Rente|Pension|Kindergeld|ChildBenefit"
        r")"
        r")",
        re.I,
    )

    # PDF scope: forms / FAQ bundles / infoblätter for private individuals.
    # Explicitly excluded from SharedDocs: Versicherung_Feuerschutz (fire-protection insurance
    # tax — corporate/insurer topic), EOP_BOP (portal registration for businesses), and IBAN
    # (bank-side IBAN notification templates — not taxpayer guidance).
    pdf_allow_pattern = re.compile(
        r"bzst\.de/SharedDocs/Downloads/DE/(?!(Versicherung|EOP_BOP|IBAN/)).*\.pdf"
        r"|bzst\.de/(DE/Privatpersonen|EN/PrivateIndividuals)/.*\.pdf",
        re.I,
    )

    max_pages = 500
    max_depth = 3
