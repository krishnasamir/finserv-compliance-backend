"""Upsert chunks into PGVector; populate vector column and tsvector keyword column."""

from __future__ import annotations

import logging

from config import settings
from src.ingestion.chunker import Chunk

log = logging.getLogger(__name__)

_engine = None

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT        NOT NULL,
    framework   TEXT        NOT NULL,
    version     TEXT        NOT NULL,
    section_id  TEXT        NOT NULL,
    text        TEXT        NOT NULL,
    effective_date DATE,
    embedding   VECTOR(384),
    tsv         TSVECTOR,
    created_at  TIMESTAMP   DEFAULT NOW(),
    CONSTRAINT chunks_content_uq UNIQUE (doc_id, section_id)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
"""

_UPSERT = """
INSERT INTO chunks (doc_id, framework, version, section_id, text, effective_date, embedding, tsv)
VALUES (
    :doc_id, :framework, :version, :section_id, :text, :effective_date,
    :embedding ::vector,
    to_tsvector('english', :text)
)
ON CONFLICT (doc_id, section_id) DO UPDATE SET
    text           = EXCLUDED.text,
    framework      = EXCLUDED.framework,
    version        = EXCLUDED.version,
    embedding      = EXCLUDED.embedding,
    tsv            = EXCLUDED.tsv,
    effective_date = EXCLUDED.effective_date;
"""


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _engine


def ensure_schema() -> None:
    """Create the chunks table and indexes if they do not already exist."""
    from sqlalchemy import text
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(text(_DDL))


def load(chunks: list[Chunk]) -> None:
    """Upsert chunks into Postgres/pgvector.

    Populates both the vector column (for dense search) and the tsvector column
    (for keyword/full-text search).
    """
    if not chunks:
        return

    from sqlalchemy import text
    ensure_schema()
    engine = _get_engine()

    rows = []
    for c in chunks:
        if c.embedding is None:
            raise ValueError(f"Chunk {c.doc_id}/{c.section_id} has no embedding — call embed() first.")
        rows.append({
            "doc_id": c.doc_id,
            "framework": c.framework,
            "version": c.version,
            "section_id": c.section_id,
            "text": c.text,
            "effective_date": c.effective_date,
            "embedding": str(c.embedding),
        })

    with engine.begin() as conn:
        conn.execute(text(_UPSERT), rows)

    log.info("Loaded %d chunks into PGVector.", len(chunks))
