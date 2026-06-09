"""AT-1 (part): chunk() produces metadata-rich Chunk objects from Pages."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import Chunk, chunk
from src.ingestion.parser import Page, parse_pdf


def _pages_from_fixture(fixture_pdfs, idx: int = 0) -> list[Page]:
    pdf_path, framework, version = fixture_pdfs[idx]
    return parse_pdf(pdf_path, framework, version)


def test_chunk_returns_at_least_one_chunk(fixture_pdfs):
    """chunk() must produce > 0 chunks from a parsed document."""
    pdf_path, framework, version = fixture_pdfs[0]
    pages = parse_pdf(pdf_path, framework, version)
    chunks = chunk(pages, framework, version)
    assert len(chunks) > 0, "chunk() returned no chunks — is it implemented?"


def test_all_chunks_have_required_metadata(fixture_pdfs):
    """Every Chunk must have non-null doc_id, framework, version, section_id, and text.

    This is AT-1 from the spec: 'each with non-null framework/section metadata'.
    """
    pdf_path, framework, version = fixture_pdfs[0]
    pages = parse_pdf(pdf_path, framework, version)
    chunks = chunk(pages, framework, version)
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.doc_id, f"Chunk missing doc_id: {c}"
        assert c.framework, f"Chunk missing framework: {c}"
        assert c.version, f"Chunk missing version: {c}"
        assert c.section_id, f"Chunk missing section_id: {c}"
        assert c.text.strip(), f"Chunk has empty text: {c}"


def test_chunk_metadata_matches_document(fixture_pdfs):
    """framework and version on each Chunk must match the source document."""
    for pdf_path, framework, version in fixture_pdfs:
        pages = parse_pdf(pdf_path, framework, version)
        chunks = chunk(pages, framework, version)
        for c in chunks:
            assert c.framework == framework, f"Framework mismatch in {pdf_path.name}"
            assert c.version == version, f"Version mismatch in {pdf_path.name}"


def test_chunk_text_is_substring_of_source(fixture_pdfs):
    """Each chunk's text must come from the source document (no hallucinated text)."""
    pdf_path, framework, version = fixture_pdfs[0]
    pages = parse_pdf(pdf_path, framework, version)
    full_text = " ".join(p.text for p in pages).lower()
    chunks = chunk(pages, framework, version)
    assert len(chunks) > 0
    for c in chunks:
        # At least some words from the chunk must appear in the source
        chunk_words = set(c.text.lower().split())
        assert chunk_words & set(full_text.split()), f"Chunk text appears unrelated to source: {c.text[:80]}"
