"""Clause/section-aware chunking of parsed regulatory pages."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from typing import Optional

from config import settings

# Matches section/article/chapter headings and numbered clauses like "3.2.1 Heading"
_SECTION_RE = re.compile(
    r"(?m)^(?:"
    r"(?:Section|Article|Chapter|Part|Para(?:graph)?)\s+[\d.]+[A-Za-z]?"
    r"|"
    r"\d+(?:\.\d+){1,3}\s+[A-Z][A-Za-z]"
    r")"
)

# Approximate: 1 BGE token ≈ 4 characters for English regulatory text
_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A tokenised, metadata-rich slice of a regulatory document."""

    doc_id: str
    framework: str
    version: str
    section_id: str
    text: str
    effective_date: Optional[datetime.date] = None
    embedding: Optional[list[float]] = field(default=None, repr=False)


def chunk(pages: list, framework: str, version: str) -> list[Chunk]:
    """Split pages into chunks of ~CHUNK_TARGET_TOKENS with CHUNK_OVERLAP_RATIO overlap.

    Tries to honour natural section/clause boundaries before applying a sliding
    window so every chunk stays within one logical section where possible.
    """
    target_chars = settings.chunk_target_tokens * _CHARS_PER_TOKEN
    overlap_chars = int(target_chars * settings.chunk_overlap_ratio)
    step_chars = target_chars - overlap_chars

    if not pages:
        return []

    # Infer doc_id from the first page
    doc_id = pages[0].doc_id

    # Concatenate all page text preserving page boundaries
    full_text = "\n\n".join(p.text for p in pages)

    # Find section boundaries
    splits = [m.start() for m in _SECTION_RE.finditer(full_text)]
    # Add start and end sentinels
    if not splits or splits[0] != 0:
        splits.insert(0, 0)
    splits.append(len(full_text))

    chunks: list[Chunk] = []
    chunk_index = 0

    for seg_idx in range(len(splits) - 1):
        seg_start = splits[seg_idx]
        seg_end = splits[seg_idx + 1]
        segment = full_text[seg_start:seg_end].strip()

        if not segment:
            continue

        # Derive a section label from the first line of the segment
        first_line = segment.split("\n", 1)[0].strip()
        # Normalise: keep first 60 chars, replace whitespace with underscores
        section_label = re.sub(r"\s+", "_", first_line[:60]).rstrip("_")

        # Slide a window over the segment
        pos = 0
        within_seg = 0
        while pos < len(segment):
            slice_text = segment[pos : pos + target_chars].strip()
            if not slice_text:
                break
            section_id = f"{section_label}-c{chunk_index}" if section_label else f"c{chunk_index}"
            chunks.append(Chunk(
                doc_id=doc_id,
                framework=framework,
                version=version,
                section_id=section_id,
                text=slice_text,
            ))
            chunk_index += 1
            within_seg += 1
            if len(segment) - pos <= target_chars:
                break
            pos += step_chars

    return chunks
