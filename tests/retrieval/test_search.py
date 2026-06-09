"""AT-2 & AT-3: dense/keyword/hybrid search over the loaded corpus."""

from __future__ import annotations

import pytest

from src.retrieval.search import ScoredChunk, dense_search, hybrid_search, keyword_search

# ── AT-2 — dense_search and keyword_search return results for known terms ─────

# A term that appears verbatim in the fixture corpus (basket3_cet1.pdf)
_KNOWN_TERM = "CET1"
# A term that appears verbatim in rbi_kyc.pdf
_KNOWN_SECTION = "Section 3.2.1"


def test_dense_search_returns_results_for_known_term(loaded_corpus):
    """AT-2: dense_search returns ≥ 1 result for a term known to be in the corpus."""
    results = dense_search(_KNOWN_TERM, k=5)
    assert len(results) > 0, (
        f"dense_search('{_KNOWN_TERM}') returned no results — is it implemented?"
    )
    assert all(isinstance(r, ScoredChunk) for r in results)


def test_keyword_search_returns_results_for_known_term(loaded_corpus):
    """AT-2: keyword_search returns ≥ 1 result for a term known to be in the corpus."""
    results = keyword_search(_KNOWN_TERM, k=5)
    assert len(results) > 0, (
        f"keyword_search('{_KNOWN_TERM}') returned no results — is it implemented?"
    )
    assert all(isinstance(r, ScoredChunk) for r in results)


def test_dense_search_respects_k_limit(loaded_corpus):
    """dense_search must return at most k results."""
    k = 3
    results = dense_search(_KNOWN_TERM, k=k)
    assert len(results) <= k, f"dense_search returned {len(results)} results but k={k}"


def test_keyword_search_respects_k_limit(loaded_corpus):
    """keyword_search must return at most k results."""
    k = 3
    results = keyword_search(_KNOWN_TERM, k=k)
    assert len(results) <= k, f"keyword_search returned {len(results)} results but k={k}"


def test_search_results_have_positive_scores(loaded_corpus):
    """All ScoredChunk scores must be positive (retrieval score, not distance)."""
    for fn, name in ((dense_search, "dense_search"), (keyword_search, "keyword_search")):
        results = fn(_KNOWN_TERM, k=5)
        for r in results:
            assert r.score > 0, f"{name} returned a non-positive score: {r.score}"


# ── AT-3 — hybrid_search covers gaps of each individual mode ──────────────────

def test_hybrid_search_returns_results(loaded_corpus):
    """AT-3 (baseline): hybrid_search must return ≥ 1 result for a known-corpus query."""
    results = hybrid_search(_KNOWN_TERM, k=10)
    assert len(results) > 0, "hybrid_search returned no results — is it implemented?"


def test_hybrid_covers_semantic_gap(loaded_corpus):
    """AT-3 (semantic gap): hybrid finds relevant chunks even when keyword-only would miss them.

    Uses a paraphrase of corpus content — no exact token match — so keyword
    search is expected to return fewer results than hybrid.
    """
    # Paraphrase of Basel III CET1 content; no exact word match with corpus text
    semantic_q = "lenders must hold tier-one equity as a fraction of weighted exposures"

    hybrid = hybrid_search(semantic_q, k=10)
    keyword = keyword_search(semantic_q, k=10)

    # Hybrid must return at least as many results as keyword alone
    assert len(hybrid) >= len(keyword), (
        "hybrid_search returned fewer results than keyword_search on a semantic query"
    )
    # And hybrid must actually find something via the dense path
    assert len(hybrid) > 0, (
        "hybrid_search found nothing for a semantic query — dense component not implemented?"
    )


def test_hybrid_covers_exact_term_gap(loaded_corpus):
    """AT-3 (keyword gap): hybrid finds exact section references that dense-only might miss.

    Section numbers are arbitrary tokens; dense (semantic) search may rank them
    low. Keyword search must find them, and hybrid must preserve that coverage.
    """
    # Exact section reference present verbatim in the corpus
    exact_q = "Section 3.2.1"

    hybrid = hybrid_search(exact_q, k=10)
    dense = dense_search(exact_q, k=10)

    # Hybrid must return at least as many results as dense alone
    assert len(hybrid) >= len(dense), (
        "hybrid_search returned fewer results than dense_search on an exact-term query"
    )
    # And hybrid must find the exact term via the keyword path
    assert len(hybrid) > 0, (
        "hybrid_search found nothing for an exact section reference — keyword component not implemented?"
    )


def test_hybrid_deduplicates_results(loaded_corpus):
    """AT-3: hybrid_search must not return duplicate chunks (same doc_id + section_id)."""
    results = hybrid_search(_KNOWN_TERM, k=20)
    if not results:
        pytest.skip("hybrid_search not yet implemented — skip dedup check")
    ids = [(r.chunk.doc_id, r.chunk.section_id) for r in results]
    assert len(ids) == len(set(ids)), "hybrid_search returned duplicate chunks"
