"""Add BGE-small embedding vectors to chunks (local, sentence-transformers)."""

from __future__ import annotations

import logging

from config import settings
from src.ingestion.chunker import Chunk

log = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s …", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed(chunks: list[Chunk]) -> list[Chunk]:
    """Populate the embedding field on each chunk in-place and return the list.

    Uses BAAI/bge-small-en-v1.5 via sentence-transformers (local, open-source).
    """
    if not chunks:
        return chunks

    model = _get_model()
    texts = [c.text for c in chunks]
    log.info("Embedding %d chunks …", len(texts))
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    for c, vec in zip(chunks, vectors):
        c.embedding = vec.tolist()
    return chunks
