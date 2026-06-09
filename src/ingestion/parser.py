

"""PDF parsing — extract text per page, preserving section/clause structure."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class Page:
    """A single page extracted from a regulatory PDF."""

    def __init__(
        self,
        page_number: int,
        text: str,
        doc_id: str,
        framework: str,
        version: str,
    ) -> None:
        self.page_number = page_number
        self.text = text
        self.doc_id = doc_id
        self.framework = framework
        self.version = version

    def __repr__(self) -> str:
        return f"Page(doc_id={self.doc_id!r}, page={self.page_number}, chars={len(self.text)})"


def parse_pdf(path: Path | str, framework: str, version: str) -> list[Page]:
    """Extract text per page from a regulatory PDF.

    Falls back gracefully on malformed pages (logs a warning, continues).
    Uses PyMuPDF (fitz) — open-source; replaces Adobe PDF Services.
    """
    import fitz  # PyMuPDF

    path = Path(path)
    doc_id = path.stem
    pages: list[Page] = []

    try:
        with fitz.open(str(path)) as doc:
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text("text")
                    text = text.strip()
                    if not text:
                        log.debug("Page %d of %s has no extractable text; skipping.", page_num + 1, path.name)
                        continue
                    pages.append(Page(
                        page_number=page_num + 1,
                        text=text,
                        doc_id=doc_id,
                        framework=framework,
                        version=version,
                    ))
                except Exception as exc:
                    log.warning("Skipping malformed page %d in %s: %s", page_num + 1, path.name, exc)
    except Exception as exc:
        log.error("Failed to open PDF %s: %s", path, exc)
        raise

    return pages
