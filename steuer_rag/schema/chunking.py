"""Deterministic, character-based chunker.

Same approach as the reference design: byte-stable offsets, model-agnostic, prefers natural breaks
(`\\n\\n`, `. `, `\\n`, ` `) within the last `look_back` characters of the window so we don't slice
words or sentences. Returns rich `DocumentChunk` rows linked back to a `DocumentCore`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from steuer_rag.schema.models import DocumentChunk, DocumentCore


@dataclass(slots=True)
class TextChunk:
    content: str
    start_char: int
    end_char: int
    chunk_index: int


def _find_break(text: str, lo: int, hi: int) -> int:
    """Find the best natural-break offset in `text[lo:hi]`. Returns `hi` if none found."""
    window = text[lo:hi]
    for sep in ("\n\n", ". ", "\n", " "):
        idx = window.rfind(sep)
        if idx > 0:
            return lo + idx + len(sep)
    return hi


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chunk_chars: int = 200,
    look_back: int = 200,
) -> list[TextChunk]:
    """Split `text` into overlapping chunks preserving natural sentence/paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    n = len(text)
    if n == 0:
        return []
    if n <= chunk_size:
        return [TextChunk(text, 0, n, 0)] if n >= min_chunk_chars else []

    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    while start < n:
        target_end = min(start + chunk_size, n)
        if target_end < n:
            end = _find_break(text, max(target_end - look_back, start + 1), target_end)
        else:
            end = target_end
        piece = text[start:end].strip()
        if len(piece) >= min_chunk_chars:
            chunks.append(TextChunk(piece, start, end, idx))
            idx += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunks_for_document(
    doc: DocumentCore,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chunk_chars: int = 200,
    strategy_version: str = "v1",
) -> list[DocumentChunk]:
    """Turn a `DocumentCore` into a list of `DocumentChunk` rows. Content-addressable chunk IDs."""
    pieces = chunk_text(
        doc.content,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_chars=min_chunk_chars,
    )
    out: list[DocumentChunk] = []
    for p in pieces:
        content_hash = hashlib.sha256(p.content.encode("utf-8")).hexdigest()
        chunk_id = hashlib.sha256(
            f"{doc.doc_id}|{p.chunk_index}|{content_hash}".encode("utf-8")
        ).hexdigest()[:16]
        out.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                document_key=doc.document_key,
                source=doc.source,
                url=doc.url,
                doc_title=doc.doc_title,
                section=doc.section,
                language=doc.language,
                content=p.content,
                chunk_index=p.chunk_index,
                start_char=p.start_char,
                end_char=p.end_char,
                content_chars=len(p.content),
                content_hash=content_hash[:16],
                chunk_strategy_version=strategy_version,
            )
        )
    return out
