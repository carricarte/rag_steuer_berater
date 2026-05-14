"""Typed document + chunk models.

Mirrors the `DocumentCore`/`DocumentChunk` pattern from the reference design — content-addressable
IDs so re-ingests are idempotent, frozen field set so downstream consumers can rely on the schema.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class Language(str, Enum):
    DE = "de"
    EN = "en"
    UNKNOWN = "unknown"


class SourceName(str, Enum):
    BMF = "bmf"
    ELSTER = "elster"
    BZST = "bzst"
    GESETZE = "gesetze"


SOURCE_BASE_URLS: dict[SourceName, str] = {
    SourceName.BMF: "https://www.bundesfinanzministerium.de",
    SourceName.ELSTER: "https://www.elster.de",
    SourceName.BZST: "https://www.bzst.de",
    SourceName.GESETZE: "https://www.gesetze-im-internet.de",
}


# --- helpers ---------------------------------------------------------------

_SHA = hashlib.sha256


def _sha(text: str) -> str:
    return _SHA(text.encode("utf-8")).hexdigest()


def detect_language(text: str) -> Language:
    """Cheap, dependency-light language detector for DE vs EN.

    Uses `langdetect` if available; falls back to a tiny heuristic on common stopwords. We only
    need to distinguish German from English — anything else maps to UNKNOWN so the retrieval layer
    can still match by content.
    """
    text = (text or "").strip()
    if len(text) < 20:
        return Language.UNKNOWN
    try:
        from langdetect import detect, DetectorFactory  # type: ignore

        DetectorFactory.seed = 0  # deterministic
        code = detect(text[:2000]).lower()
        if code.startswith("de"):
            return Language.DE
        if code.startswith("en"):
            return Language.EN
        return Language.UNKNOWN
    except Exception:
        sample = text.lower()
        de_hits = sum(sample.count(w) for w in (" der ", " die ", " und ", " ist ", " für "))
        en_hits = sum(sample.count(w) for w in (" the ", " and ", " is ", " for ", " of "))
        if de_hits > en_hits:
            return Language.DE
        if en_hits > de_hits:
            return Language.EN
        return Language.UNKNOWN


def normalize_text(text: str) -> str:
    """Normalize whitespace and line endings for deterministic chunk offsets."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- models ---------------------------------------------------------------


class DocumentCore(BaseModel):
    """One canonical document from an upstream source."""

    doc_id: str = Field(..., description="Content-addressable SHA-256 of (source, url, content_hash)")
    document_key: str = Field(..., description="Stable cross-version key (typically the URL)")
    source: SourceName
    url: HttpUrl
    doc_title: str = ""
    content: str = ""
    content_sha256: str = ""
    language: Language = Language.UNKNOWN
    doc_type: Literal["html", "pdf", "form"] = "html"
    section: str | None = None
    trust_tier: int = 1  # 1 = official .de gov source
    is_current: bool = True
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    indexed_at: datetime | None = None

    @classmethod
    def build(
        cls,
        *,
        source: SourceName,
        url: str,
        title: str,
        content: str,
        doc_type: Literal["html", "pdf", "form"] = "html",
        section: str | None = None,
    ) -> "DocumentCore":
        norm = normalize_text(content)
        content_sha = _sha(norm)
        doc_id = _sha(f"{source.value}|{url}|{content_sha}")[:24]
        return cls(
            doc_id=doc_id,
            document_key=url,
            source=source,
            url=url,  # type: ignore[arg-type]
            doc_title=title.strip(),
            content=norm,
            content_sha256=content_sha,
            language=detect_language(norm),
            doc_type=doc_type,
            section=section,
        )


class DocumentChunk(BaseModel):
    """A chunk of a `DocumentCore`. This is what the retrieval index sees."""

    chunk_id: str
    doc_id: str
    document_key: str
    source: SourceName
    url: HttpUrl
    doc_title: str
    doc_type: Literal["html", "pdf", "form"] = "html"
    section: str | None = None
    language: Language
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    content_chars: int
    content_hash: str
    chunk_strategy_version: str = "v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_metadata(self) -> dict:
        """Flatten to a plain dict for vector-store metadata (Chroma rejects nested types)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "document_key": self.document_key,
            "source": self.source.value,
            "url": str(self.url),
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "section": self.section or "",
            "language": self.language.value,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "content_chars": self.content_chars,
            "content_hash": self.content_hash,
            "chunk_strategy_version": self.chunk_strategy_version,
            "created_at": self.created_at.isoformat(),
        }
