"""AT-1 (part): parse_pdf extracts pages from a regulatory PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.parser import Page, parse_pdf


def test_parse_pdf_returns_at_least_one_page(fixture_pdfs):
    """parse_pdf must return a non-empty list of Pages for any valid PDF."""
    pdf_path, framework, version = fixture_pdfs[0]
    pages = parse_pdf(pdf_path, framework, version)
    assert len(pages) > 0, "parse_pdf returned no pages — is it implemented?"


def test_all_pages_have_nonempty_text(fixture_pdfs):
    """Every Page must contain extracted text (no silently blank pages)."""
    pdf_path, framework, version = fixture_pdfs[0]
    pages = parse_pdf(pdf_path, framework, version)
    assert len(pages) > 0
    for page in pages:
        assert isinstance(page, Page)
        assert page.text.strip(), f"Page {page.page_number} has empty text"


def test_pages_carry_doc_metadata(fixture_pdfs):
    """framework and version must be propagated to every Page."""
    pdf_path, framework, version = fixture_pdfs[0]
    pages = parse_pdf(pdf_path, framework, version)
    assert len(pages) > 0
    for page in pages:
        assert page.framework == framework
        assert page.version == version
        assert page.doc_id, "doc_id must be non-empty"


def test_parse_pdf_handles_all_fixture_documents(fixture_pdfs):
    """parse_pdf must succeed (not crash) for each document in the fixture corpus."""
    for pdf_path, framework, version in fixture_pdfs:
        pages = parse_pdf(pdf_path, framework, version)
        assert isinstance(pages, list), f"Expected list, got {type(pages)} for {pdf_path.name}"
        assert len(pages) > 0, f"No pages extracted from {pdf_path.name}"
