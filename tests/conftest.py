"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

from pathlib import Path

import pytest


# ── Minimal PDF builder (no external deps) ───────────────────────────────────

def _make_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF containing *text* (no external dependencies).

    Computes xref offsets programmatically so the PDF is valid enough for PyMuPDF
    to parse once the implementation is written.
    """
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 50 700 Td ({safe}) Tj ET".encode("latin-1")
    slen = len(stream)

    parts: list[bytes] = []

    def _pos() -> int:
        return sum(len(p) for p in parts)

    parts.append(b"%PDF-1.4\n")

    o1 = _pos()
    parts.append(b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n")
    o2 = _pos()
    parts.append(b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n")
    o3 = _pos()
    parts.append(
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources <</Font <</F1 4 0 R>>>> /Contents 5 0 R>>\nendobj\n"
    )
    o4 = _pos()
    parts.append(b"4 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n")
    o5 = _pos()
    parts.append(
        f"5 0 obj\n<</Length {slen}>>\nstream\n".encode("latin-1")
        + stream
        + b"\nendstream\nendobj\n"
    )

    xref_pos = _pos()
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for off in (o1, o2, o3, o4, o5):
        xref += f"{off:010d} 00000 n \n".encode("ascii")
    parts.append(xref)
    parts.append(
        f"trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return b"".join(parts)


# ── Fixture corpus — three minimal regulatory documents ───────────────────────

_CORPUS: list[tuple[str, str, str]] = [
    # (filename_stem, framework, version)
    (
        "basel3_cet1",
        "Basel III",
        "2023",
    ),
    (
        "rbi_kyc",
        "RBI-KYC",
        "2024",
    ),
    (
        "mifid_market",
        "MiFID II",
        "2023",
    ),
]

_CORPUS_TEXT: dict[str, str] = {
    "basel3_cet1": (
        "Basel III capital adequacy framework. "
        "CET1 minimum ratio 4.5 percent of risk-weighted assets. "
        "Section 3.2.1 defines the Common Equity Tier 1 requirement. "
        "Banks must maintain adequate capital buffers at all times."
    ),
    "rbi_kyc": (
        "Reserve Bank of India Know Your Customer KYC guidelines. "
        "Section 4.1 customer due diligence requirements for financial institutions. "
        "Compliance with anti-money laundering AML policies is mandatory."
    ),
    "mifid_market": (
        "MiFID II market risk capital requirements for investment firms. "
        "Section 2.7 specifies the minimum threshold for trading book positions. "
        "Investment firms must report risk exposure on a daily basis."
    ),
}


@pytest.fixture(scope="session")
def fixture_pdfs(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[Path, str, str]]:
    """Write three minimal regulatory PDFs to a temp directory.

    Returns a list of (pdf_path, framework, version) tuples.
    Tests use these instead of the real PDFs in data/ so the test suite is
    self-contained.
    """
    tmp = tmp_path_factory.mktemp("fixture_pdfs")
    result = []
    for stem, framework, version in _CORPUS:
        path = tmp / f"{stem}.pdf"
        path.write_bytes(_make_pdf(_CORPUS_TEXT[stem]))
        result.append((path, framework, version))
    return result


@pytest.fixture(scope="session")
def loaded_corpus(fixture_pdfs: list[tuple[Path, str, str]]):
    """Run the full ingestion pipeline on fixture PDFs and return all chunks.

    Scope is session so the (expensive) pipeline runs once per test session.
    With stubs, returns []. With real implementation, returns populated Chunks.
    """
    from src.ingestion.chunker import chunk
    from src.ingestion.embedder import embed
    from src.ingestion.loader import load
    from src.ingestion.parser import parse_pdf

    all_chunks = []
    for pdf_path, framework, version in fixture_pdfs:
        pages = parse_pdf(pdf_path, framework, version)
        chunks = chunk(pages, framework, version)
        embedded = embed(chunks)
        all_chunks.extend(embedded)
    load(all_chunks)
    return all_chunks


# ── Convenience fixtures ──────────────────────────────────────────────────────

@pytest.fixture()
def settings():
    """Application settings loaded from .env."""
    from config import settings as _s
    return _s


@pytest.fixture()
def sample_chunk():
    """A minimal Chunk for unit tests that don't touch the DB."""
    from src.ingestion.chunker import Chunk
    return Chunk(
        doc_id="test-doc-001",
        framework="Basel III",
        version="2023",
        section_id="S1.1",
        text="Banks must maintain a minimum CET1 ratio of 4.5% of risk-weighted assets.",
    )
