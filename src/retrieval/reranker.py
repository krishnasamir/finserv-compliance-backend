"""BGE cross-encoder reranker and context compression."""

from __future__ import annotations

import logging

from config import settings
from src.ingestion.chunker import Chunk
from src.retrieval.search import ScoredChunk

log = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        log.info("Loading reranker model %s …", settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Reorder chunks by cross-encoder relevance using BAAI/bge-reranker-base (local)."""
    if not chunks:
        return chunks

    reranker = _get_reranker()
    pairs = [(query, sc.chunk.text) for sc in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Sort by cross-encoder score descending
    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: float(x[0]),
        reverse=True,
    )
    return [ScoredChunk(chunk=sc.chunk, score=float(s)) for s, sc in ranked]


def compress(query: str, chunks: list[ScoredChunk]) -> list[Chunk]:
    """Trim each chunk to query-relevant sentences.

    Keeps sentences that share at least one content word with the query.
    Falls back to the full chunk text if no sentence matches.
    """
    import re

    query_tokens = set(re.findall(r"\w+", query.lower()))
    # Remove very common stop-words so we don't match on "the", "a", etc.
    stopwords = {
        "the", "a", "an", "and", "or", "of", "in", "to", "is", "are",
        "for", "that", "this", "it", "with", "be", "on", "at", "as",
        "by", "from", "not", "which", "its", "their", "shall", "must",
    }
    query_tokens -= stopwords

    result: list[Chunk] = []
    for sc in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", sc.chunk.text)
        relevant = [
            s for s in sentences
            if query_tokens & set(re.findall(r"\w+", s.lower()))
        ]
        compressed_text = " ".join(relevant).strip() if relevant else sc.chunk.text
        result.append(Chunk(
            doc_id=sc.chunk.doc_id,
            framework=sc.chunk.framework,
            version=sc.chunk.version,
            section_id=sc.chunk.section_id,
            text=compressed_text,
            effective_date=sc.chunk.effective_date,
        ))
    return result
