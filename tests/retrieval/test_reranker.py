"""AT-4: rerank() changes the ordering of chunks to improve precision."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.reranker import compress, rerank
from src.retrieval.search import ScoredChunk


@pytest.fixture
def deliberately_misordered_chunks():
    """Three chunks presented in reverse relevance order for a capital-ratio query.

    The chunk most relevant to 'CET1 minimum capital ratio' has the LOWEST score,
    so a real cross-encoder should promote it to the top.
    """
    irrelevant = Chunk(
        doc_id="d1",
        framework="Basel III",
        version="2023",
        section_id="S-ADMIN",
        text="General administrative procedures for compliance form submission.",
    )
    medium = Chunk(
        doc_id="d1",
        framework="Basel III",
        version="2023",
        section_id="S-RWA",
        text="Risk-weighted assets are calculated using the standardised approach.",
    )
    relevant = Chunk(
        doc_id="d1",
        framework="Basel III",
        version="2023",
        section_id="S-CET1",
        text="CET1 minimum capital ratio requirement is 4.5 percent of risk-weighted assets.",
    )
    # Worst-first ordering: highest retrieval score goes to the least relevant chunk
    return [
        ScoredChunk(chunk=irrelevant, score=0.92),
        ScoredChunk(chunk=medium, score=0.55),
        ScoredChunk(chunk=relevant, score=0.18),
    ]


def test_rerank_changes_order(deliberately_misordered_chunks):
    """AT-4: rerank must produce a different ordering than the input for this query.

    The cross-encoder should recognise that S-CET1 is most relevant and promote it.
    """
    query = "minimum CET1 capital ratio requirement for banks"
    original_ids = [sc.chunk.section_id for sc in deliberately_misordered_chunks]

    reranked = rerank(query, deliberately_misordered_chunks)

    assert len(reranked) == len(deliberately_misordered_chunks), (
        "rerank must return the same number of chunks as input"
    )
    reranked_ids = [sc.chunk.section_id for sc in reranked]
    assert reranked_ids != original_ids, (
        "rerank returned chunks in the same order as input — "
        "cross-encoder did not change ordering (stub passthrough?)"
    )


def test_rerank_puts_most_relevant_first(deliberately_misordered_chunks):
    """AT-4 (precision): the most relevant chunk (S-CET1) should rank first after reranking."""
    query = "minimum CET1 capital ratio requirement for banks"
    reranked = rerank(query, deliberately_misordered_chunks)
    assert reranked[0].chunk.section_id == "S-CET1", (
        f"Expected S-CET1 first after reranking, got {reranked[0].chunk.section_id}"
    )


def test_rerank_preserves_all_chunks(deliberately_misordered_chunks):
    """rerank must not drop or duplicate any chunk."""
    query = "minimum CET1 capital ratio requirement for banks"
    reranked = rerank(query, deliberately_misordered_chunks)
    original_ids = sorted(sc.chunk.section_id for sc in deliberately_misordered_chunks)
    reranked_ids = sorted(sc.chunk.section_id for sc in reranked)
    assert original_ids == reranked_ids, "rerank dropped or duplicated chunks"


def test_rerank_returns_scored_chunks(deliberately_misordered_chunks):
    """rerank output must be a list of ScoredChunk with updated scores."""
    query = "CET1 capital ratio"
    reranked = rerank(query, deliberately_misordered_chunks)
    for r in reranked:
        assert isinstance(r, ScoredChunk)
        assert isinstance(r.score, (int, float))


def test_compress_trims_to_relevant_spans(sample_chunk):
    """compress() must return Chunk objects whose text is ≤ the original length."""
    from src.retrieval.search import ScoredChunk as SC

    scored = [SC(chunk=sample_chunk, score=0.8)]
    compressed = compress("CET1 ratio", scored)
    assert len(compressed) == 1
    assert isinstance(compressed[0], Chunk)
    # Compressed text must be no longer than the original
    assert len(compressed[0].text) <= len(sample_chunk.text), (
        "compress() returned text longer than the original chunk"
    )
