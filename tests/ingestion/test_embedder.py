"""AT-1 (part): embed() attaches float vectors to each Chunk."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import Chunk
from src.ingestion.embedder import embed


def test_embed_populates_embedding_field(sample_chunk):
    """embed() must set the embedding field to a non-empty float list."""
    result = embed([sample_chunk])
    assert len(result) == 1
    c = result[0]
    assert c.embedding is not None, "embedding field must not be None after embed()"
    assert len(c.embedding) > 0, "embedding vector must be non-empty"


def test_embed_vector_has_correct_dimension(sample_chunk, settings):
    """Embedding dimension must match EMBEDDING_DIM from config (default 384 for BGE-small)."""
    result = embed([sample_chunk])
    assert result[0].embedding is not None
    assert len(result[0].embedding) == settings.embedding_dim, (
        f"Expected dim {settings.embedding_dim}, got {len(result[0].embedding)}"
    )


def test_embed_preserves_all_other_fields(sample_chunk):
    """embed() must not alter any metadata field — only sets embedding."""
    original_text = sample_chunk.text
    original_doc_id = sample_chunk.doc_id
    result = embed([sample_chunk])
    c = result[0]
    assert c.text == original_text
    assert c.doc_id == original_doc_id


def test_embed_returns_same_count(fixture_pdfs):
    """embed() must return exactly as many chunks as it receives."""
    from src.ingestion.chunker import chunk
    from src.ingestion.parser import parse_pdf

    pdf_path, framework, version = fixture_pdfs[0]
    pages = parse_pdf(pdf_path, framework, version)
    chunks = chunk(pages, framework, version)
    assert len(chunks) > 0, "Need chunks to test embedder (parser/chunker not yet implemented)"
    embedded = embed(chunks)
    assert len(embedded) == len(chunks)
