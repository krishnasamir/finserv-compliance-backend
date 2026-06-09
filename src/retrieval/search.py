"""Dense, keyword, and hybrid retrieval over the PGVector store."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings
from src.ingestion.chunker import Chunk

log = logging.getLogger(__name__)

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model for retrieval: %s", settings.embedding_model)
        _embed_model = SentenceTransformer(settings.embedding_model)
    return _embed_model


def _embed_query(query: str) -> list[float]:
    model = _get_embed_model()
    vec = model.encode([query], normalize_embeddings=True)[0]
    return vec.tolist()


def _get_engine():
    from sqlalchemy import create_engine
    return create_engine(settings.database_url, pool_pre_ping=True)


@dataclass
class ScoredChunk:
    """A retrieved chunk with its retrieval score."""

    chunk: Chunk
    score: float


def _row_to_scored(row) -> ScoredChunk:
    c = Chunk(
        doc_id=row.doc_id,
        framework=row.framework,
        version=row.version,
        section_id=row.section_id,
        text=row.text,
    )
    return ScoredChunk(chunk=c, score=float(row.score))


def dense_search(query: str, k: int) -> list[ScoredChunk]:
    """PGVector cosine top-k search (1 − cosine_distance, so higher = more similar)."""
    from sqlalchemy import text

    embedding = _embed_query(query)
    sql = text("""
        SELECT doc_id, framework, version, section_id, text,
               1 - (embedding <=> :emb ::vector) AS score
        FROM chunks
        ORDER BY embedding <=> :emb ::vector
        LIMIT :k
    """)
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql, {"emb": str(embedding), "k": k}).fetchall()
    return [_row_to_scored(r) for r in rows]


def keyword_search(query: str, k: int) -> list[ScoredChunk]:
    """Postgres full-text (tsvector) top-k search."""
    from sqlalchemy import text

    sql = text("""
        SELECT doc_id, framework, version, section_id, text,
               ts_rank_cd(tsv, query) AS score
        FROM chunks,
             plainto_tsquery('english', :query) query
        WHERE tsv @@ query
        ORDER BY score DESC
        LIMIT :k
    """)
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql, {"query": query, "k": k}).fetchall()
    return [_row_to_scored(r) for r in rows]


def reciprocal_rank_fusion(
    lists: list[list[ScoredChunk]],
    k: int,
    rrf_k: int,
) -> list[ScoredChunk]:
    """Merge multiple ranked lists with Reciprocal Rank Fusion.

    rrf_k is the RRF constant (default 60 per config) that dampens high-rank
    advantage. Result is sorted by descending fusion score and limited to k items.

    Note: the returned scores are RRF fusion scores (~1/rrf_k), NOT cosine
    similarities.  They must NOT be compared against retrieval_score_threshold.
    """
    scores: dict[tuple[str, str], float] = {}
    # Keep one ScoredChunk per (doc_id, section_id) — whichever has the highest
    # individual score, for the final result construction
    best: dict[tuple[str, str], ScoredChunk] = {}

    for ranked_list in lists:
        for rank, sc in enumerate(ranked_list, start=1):
            key = (sc.chunk.doc_id, sc.chunk.section_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in best or sc.score > best[key].score:
                best[key] = sc

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [
        ScoredChunk(chunk=best[key].chunk, score=rrf_score)
        for key, rrf_score in ordered
    ]


def hybrid_search(query: str, k: int) -> list[ScoredChunk]:
    """Run dense + keyword, fuse with reciprocal rank fusion, deduplicate."""
    dense = dense_search(query, k)
    keyword = keyword_search(query, k)
    return reciprocal_rank_fusion([dense, keyword], k=k, rrf_k=settings.hybrid_rrf_k)
