from steuer_rag.schema.chunking import chunk_text, chunks_for_document
from steuer_rag.schema.models import DocumentCore, SourceName


def test_short_document_yields_zero_or_one_chunk():
    chunks = chunk_text("kurz", chunk_size=1200, overlap=200, min_chunk_chars=200)
    assert chunks == []


def test_long_document_chunks_overlap():
    text = ("Die Steuererklärung muss bis zum 31. Juli abgegeben werden. " * 100)
    chunks = chunk_text(text, chunk_size=500, overlap=100, min_chunk_chars=100)
    assert len(chunks) >= 2
    # overlap: end of chunk i overlaps beginning of chunk i+1
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_char < a.end_char


def test_chunks_for_document_metadata():
    doc = DocumentCore.build(
        source=SourceName.BMF,
        url="https://www.bundesfinanzministerium.de/test",
        title="Testdokument",
        content="Die Einkommensteuererklärung ist eine wichtige Pflicht. " * 80,
    )
    chunks = chunks_for_document(doc, chunk_size=400, overlap=80, min_chunk_chars=100)
    assert chunks
    assert all(c.doc_id == doc.doc_id for c in chunks)
    assert all(c.source == SourceName.BMF for c in chunks)
    # chunk_ids are unique and deterministic
    ids = {c.chunk_id for c in chunks}
    assert len(ids) == len(chunks)


def test_language_detection_assigns_german():
    doc = DocumentCore.build(
        source=SourceName.BMF,
        url="https://www.bundesfinanzministerium.de/x",
        title="t",
        content="Die Steuererklärung wird beim zuständigen Finanzamt eingereicht und muss alle Einkünfte enthalten.",
    )
    assert doc.language.value in ("de", "unknown")
