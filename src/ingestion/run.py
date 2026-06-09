"""CLI entrypoint: ingest all PDFs in DATA_DIR.

Usage:
    python -m src.ingestion.run

Reads DATA_DIR from config (.env).  For each PDF it finds, derives the
framework/version from a mapping table (falls back to filename-based defaults).
Prints a progress summary and a sample of stored metadata on completion.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingestion.run")


# Map filename stems to (framework, version) metadata.
# Add entries here as new regulatory PDFs are placed in DATA_DIR.
_KNOWN_DOCS: dict[str, tuple[str, str]] = {
    # Basel III / BCBS documents
    "bcbs211":   ("Basel III",  "2011"),
    "bcbs221":   ("Basel III",  "2012"),
    "d424":      ("Basel III",  "2019"),
    "defcap_b3": ("Basel III",  "2010"),
    # RBI Master Circulars
    "KYC09062025":                        ("RBI-KYC", "2025"),
    "MD18KYCF6E92C82E1E1419D87323E3869BC9F13": ("RBI-KYC", "2024"),
}

_DEFAULT_FRAMEWORK = "Regulatory"
_DEFAULT_VERSION   = "unknown"


def _resolve_metadata(stem: str) -> tuple[str, str]:
    """Return (framework, version) for a PDF stem, falling back to defaults."""
    return _KNOWN_DOCS.get(stem, (_DEFAULT_FRAMEWORK, _DEFAULT_VERSION))


def ingest_directory(data_dir: Path) -> list:
    """Ingest every PDF in *data_dir* and return all loaded Chunk objects."""
    from src.ingestion.chunker import chunk
    from src.ingestion.embedder import embed
    from src.ingestion.loader import load
    from src.ingestion.parser import parse_pdf

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        log.warning("No PDFs found in %s", data_dir)
        return []

    log.info("Found %d PDF(s) in %s", len(pdfs), data_dir)
    all_chunks = []

    for pdf_path in pdfs:
        framework, version = _resolve_metadata(pdf_path.stem)
        log.info("→ %s  [%s %s]", pdf_path.name, framework, version)

        pages = parse_pdf(pdf_path, framework, version)
        if not pages:
            log.warning("  No pages extracted from %s — skipping.", pdf_path.name)
            continue
        log.info("  Parsed %d pages.", len(pages))

        chunks = chunk(pages, framework, version)
        log.info("  Chunked into %d chunk(s).", len(chunks))

        embed(chunks)
        log.info("  Embedded %d chunk(s).", len(chunks))

        load(chunks)
        log.info("  Loaded %d chunk(s) into PGVector.", len(chunks))
        all_chunks.extend(chunks)

    return all_chunks


def main() -> None:
    from config import settings

    data_dir = settings.data_dir.expanduser().resolve()
    if not data_dir.exists():
        log.error("DATA_DIR %s does not exist.", data_dir)
        sys.exit(1)

    log.info("Starting ingestion from %s", data_dir)
    chunks = ingest_directory(data_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Ingestion complete: {len(chunks)} chunk(s) loaded")
    print(f"{'='*60}")

    if not chunks:
        print("  No chunks produced.")
        return

    # Counts by document
    from collections import Counter
    by_doc = Counter(c.doc_id for c in chunks)
    print(f"\n  Chunks by document:")
    for doc_id, count in sorted(by_doc.items()):
        print(f"    {doc_id:<45} {count:>5} chunk(s)")

    # Sample of metadata
    print(f"\n  Sample of stored chunk metadata (first 5):")
    header = f"  {'doc_id':<25} {'framework':<12} {'version':<8} {'section_id':<50} chars"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in chunks[:5]:
        sid = c.section_id[:48] + ".." if len(c.section_id) > 50 else c.section_id
        print(f"  {c.doc_id:<25} {c.framework:<12} {c.version:<8} {sid:<50} {len(c.text)}")

    print()


if __name__ == "__main__":
    main()
