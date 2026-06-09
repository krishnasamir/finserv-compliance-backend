"""AT-5 & AT-6: answer() returns cited responses and handles error paths correctly."""

from __future__ import annotations

import pytest

from src.exceptions import EmptyQueryError
from src.rag.answer import Answer, Citation, answer

# A question whose answer appears in the fixture corpus (Basel III CET1 content)
_KNOWN_QUESTION = "What is the minimum CET1 capital ratio requirement under Basel III?"

# A query designed to find nothing in the corpus
_NO_MATCH_QUERY = "xyzzy_nonexistent_regulation_clause_99999"


# ── AT-5 — answer() returns text + citations ─────────────────────────────────

def test_answer_returns_answer_type(loaded_corpus):
    """answer() must return an Answer object, not raise."""
    result = answer(_KNOWN_QUESTION)
    assert isinstance(result, Answer), f"Expected Answer, got {type(result)}"


def test_answer_returns_nonempty_text(loaded_corpus):
    """AT-5: answer() must return non-empty text for a known question."""
    result = answer(_KNOWN_QUESTION)
    assert result.text.strip(), (
        "answer() returned an empty text — LLM generation not yet implemented"
    )


def test_answer_returns_at_least_one_citation(loaded_corpus):
    """AT-5: every answer must carry ≥ 1 source citation."""
    result = answer(_KNOWN_QUESTION)
    assert len(result.citations) >= 1, (
        "answer() returned no citations — citation extraction not yet implemented"
    )


def test_answer_citation_has_required_fields(loaded_corpus):
    """AT-5: each Citation must have non-empty doc_id, section_id, and version."""
    result = answer(_KNOWN_QUESTION)
    assert len(result.citations) >= 1
    for cit in result.citations:
        assert isinstance(cit, Citation)
        assert cit.doc_id, f"Citation missing doc_id: {cit}"
        assert cit.section_id, f"Citation missing section_id: {cit}"
        assert cit.version, f"Citation missing version: {cit}"


def test_answer_citation_resolves_to_ingested_chunk(loaded_corpus):
    """AT-5: every citation must point to a chunk that was actually ingested.

    This guards against hallucinated source references.
    """
    if not loaded_corpus:
        pytest.skip("loaded_corpus is empty — ingestion pipeline not yet implemented")

    ingested_keys = {(c.doc_id, c.section_id) for c in loaded_corpus}
    result = answer(_KNOWN_QUESTION)
    assert len(result.citations) >= 1

    for cit in result.citations:
        assert (cit.doc_id, cit.section_id) in ingested_keys, (
            f"Citation ({cit.doc_id}, {cit.section_id}) does not resolve to any ingested chunk. "
            "Possible hallucinated source."
        )


# ── AT-6 — error paths ───────────────────────────────────────────────────────

def test_empty_string_raises_empty_query_error():
    """AT-6: answer('') must raise EmptyQueryError, not return an empty Answer."""
    with pytest.raises(EmptyQueryError):
        answer("")


def test_whitespace_only_raises_empty_query_error():
    """AT-6: whitespace-only queries are also invalid and must raise EmptyQueryError."""
    with pytest.raises(EmptyQueryError):
        answer("   \t\n  ")


def test_no_match_query_does_not_crash(loaded_corpus):
    """AT-6: a query with no corpus match must return an Answer, not raise."""
    result = answer(_NO_MATCH_QUERY)
    assert isinstance(result, Answer), (
        f"answer() raised or returned wrong type for a no-match query: {type(result)}"
    )


def test_no_match_returns_insufficient_context_signal(loaded_corpus):
    """AT-6: when retrieval finds nothing, the answer must signal 'insufficient context'.

    The response must NOT fabricate content. It should contain one of the recognised
    uncertainty phrases so callers know to escalate to human review.
    """
    result = answer(_NO_MATCH_QUERY)
    # Must have text (not silently empty)
    assert result.text, "answer() returned empty text for a no-match query"
    # Must signal uncertainty — never fabricate
    uncertainty_phrases = {
        "insufficient", "not enough", "no relevant", "cannot", "human review",
        "context", "unable", "no information",
    }
    lower = result.text.lower()
    assert any(phrase in lower for phrase in uncertainty_phrases), (
        f"No-match answer does not signal insufficient context.\n"
        f"Got: {result.text!r}\n"
        f"Expected one of: {uncertainty_phrases}"
    )


def test_no_match_returns_empty_citations(loaded_corpus):
    """AT-6: a no-match answer must have zero citations (nothing to cite)."""
    result = answer(_NO_MATCH_QUERY)
    assert result.citations == [], (
        "answer() attached citations to a no-match response — possible hallucination"
    )


# ── Regression: gate scale-separation ────────────────────────────────────────

def test_tier1_dense_score_exceeds_threshold(loaded_corpus, settings):
    """Raw cosine for 'Tier 1 capital adequacy Basel III' must exceed the threshold.

    RRF scores for this query are ~0.03 — below the 0.3 threshold.  If the gate
    ever switches to comparing RRF scores, this test fails first and points at
    the root cause before test_tier1_query_returns_cited_answer fires.
    """
    from src.retrieval.search import dense_search
    results = dense_search("Tier 1 capital adequacy Basel III", k=1)
    assert results, "dense_search returned no results for a known-corpus query"
    assert results[0].score >= settings.retrieval_score_threshold, (
        f"cosine={results[0].score:.3f} < threshold={settings.retrieval_score_threshold:.2f}. "
        "RRF scores (~0.03) would also fail this check — the gate must use raw cosine."
    )


def test_tier1_query_returns_cited_answer(loaded_corpus):
    """Regression: 'Tier 1 capital adequacy Basel III' must return a cited answer.

    Top chunk sits at raw cosine >> 0.3 but RRF score ≈ 0.03.  Any gate that
    compares against RRF scores would incorrectly refuse this valid query.
    """
    result = answer("Tier 1 capital adequacy Basel III")
    assert result.text.strip(), "answer() returned empty text for an in-corpus query"
    assert len(result.citations) >= 1, (
        "No citations for an in-corpus query — the no-match gate may be comparing "
        "retrieval_score_threshold (0.3) against RRF scores (~0.03) instead of raw cosine."
    )


def test_genuinely_absent_topic_refuses():
    """Regression: quantum-physics terminology absent from all financial documents
    must be refused with zero citations.
    """
    result = answer("quantum eigenstate decoherence superposition Hamiltonian")
    uncertainty_phrases = {
        "insufficient", "cannot", "unable", "no information", "no relevant",
    }
    assert any(p in result.text.lower() for p in uncertainty_phrases), (
        f"Absent-topic answer must signal uncertainty; got: {result.text!r}"
    )
    assert result.citations == [], "Absent-topic answer must carry zero citations"
