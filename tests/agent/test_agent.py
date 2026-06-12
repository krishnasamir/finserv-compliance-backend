"""AT-1 through AT-6: acceptance tests for Phase 2B — agentic compliance checker."""

from __future__ import annotations

import pytest

from src.agent.graph import assess_transaction, run_with_state
from src.agent.nodes import validate_output
from src.agent.schemas import ComplianceAssessment
from src.agent.state import AgentState

# ── Canonical test inputs ─────────────────────────────────────────────────────

_COMPLETE_TX = (
    "Cross-border payment of $2M to a non-KYC entity in a high-risk jurisdiction"
)
_AMBIGUOUS_TX = "a payment was made"
_PARSEABLE_TX = (
    "Wire transfer of $500,000 to a foreign counterparty in a sanctioned jurisdiction"
)


# ── Session-scoped caches to limit Ollama calls to once per session ───────────

@pytest.fixture(scope="session")
def complete_run(loaded_corpus):
    """Run assess_transaction on _COMPLETE_TX once; reused by all AT-1/AT-5/AT-6 tests."""
    return run_with_state(_COMPLETE_TX)


@pytest.fixture(scope="session")
def complete_assessment(complete_run):
    """ComplianceAssessment for the $2M transaction (derived from complete_run)."""
    assessment, _ = complete_run
    return assessment


@pytest.fixture(scope="session")
def complete_state(complete_run):
    """Final AgentState for the $2M transaction (derived from complete_run)."""
    _, state = complete_run
    return state


@pytest.fixture(scope="session")
def ambiguous_assessment(loaded_corpus):
    """ComplianceAssessment for the ambiguous input — cached once per session."""
    return assess_transaction(_AMBIGUOUS_TX)


# ── AT-1 — complete transaction produces a full assessed result ───────────────

def test_complete_transaction_returns_assessed(complete_assessment):
    """AT-1: $2M cross-border transaction → status='assessed'."""
    assert complete_assessment.status == "assessed", (
        f"Expected 'assessed' for complete transaction; got {complete_assessment.status!r}"
    )


def test_assessed_result_has_valid_risk_rating(complete_assessment):
    """AT-1: risk_rating must be one of the four recognised literals."""
    assert complete_assessment.status == "assessed", "Pre-condition"
    assert complete_assessment.risk_rating in {"low", "medium", "high", "critical"}, (
        f"risk_rating {complete_assessment.risk_rating!r} is not a valid literal"
    )


def test_assessed_result_has_applicable_regulations(complete_assessment):
    """AT-1: assessed result must name ≥ 1 applicable regulation."""
    assert complete_assessment.status == "assessed", "Pre-condition"
    assert len(complete_assessment.applicable_regulations) >= 1, (
        f"applicable_regulations is empty: {complete_assessment.applicable_regulations!r}"
    )


def test_assessed_result_has_required_actions(complete_assessment):
    """AT-1: assessed result must specify ≥ 1 required action."""
    assert complete_assessment.status == "assessed", "Pre-condition"
    assert len(complete_assessment.required_actions) >= 1, (
        f"required_actions is empty: {complete_assessment.required_actions!r}"
    )


def test_assessed_result_has_citations(complete_assessment):
    """AT-1: assessed result must carry ≥ 1 source citation."""
    assert complete_assessment.status == "assessed", "Pre-condition"
    assert len(complete_assessment.citations) >= 1, (
        f"citations is empty: {complete_assessment.citations!r}"
    )


# ── AT-2 — ambiguous input routes to needs_review with explanation ────────────

def test_ambiguous_input_returns_needs_review(ambiguous_assessment):
    """AT-2: under-specified transaction → status='needs_review'."""
    assert ambiguous_assessment.status == "needs_review", (
        f"Expected 'needs_review' for ambiguous input; got {ambiguous_assessment.status!r}"
    )


def test_ambiguous_input_names_missing_fields(ambiguous_assessment):
    """AT-2: needs_review result must say WHAT is missing, not just refuse silently."""
    combined = " ".join(ambiguous_assessment.required_actions).lower()
    keywords = {"amount", "counterparty", "jurisdiction", "missing", "unclear", "ambiguous"}
    assert any(kw in combined for kw in keywords), (
        f"needs_review must identify missing fields; "
        f"required_actions={ambiguous_assessment.required_actions!r}"
    )


def test_ambiguous_input_has_no_confident_rating(ambiguous_assessment):
    """AT-2: ambiguous input must not yield a confident (fabricated) risk rating."""
    assert ambiguous_assessment.status == "needs_review", "Pre-condition"
    assert ambiguous_assessment.risk_rating not in {"medium", "high", "critical"}, (
        f"Ambiguous input must not yield a confident rating; "
        f"got {ambiguous_assessment.risk_rating!r}"
    )


# ── AT-3 — below-threshold retrieval routes to needs_review ──────────────────

def test_low_confidence_retrieval_routes_to_review(monkeypatch, loaded_corpus):
    """AT-3: when dense scores are below threshold → needs_review, not a guess."""
    from src.ingestion.chunker import Chunk
    from src.retrieval.search import ScoredChunk

    dummy = ScoredChunk(
        chunk=Chunk(
            doc_id="dummy", framework="x", version="v0",
            section_id="s0", text="irrelevant text",
        ),
        score=0.01,
    )
    monkeypatch.setattr("src.retrieval.search.dense_search", lambda q, k: [dummy])
    monkeypatch.setattr("src.retrieval.search.keyword_search", lambda q, k: [])

    result = assess_transaction(_PARSEABLE_TX)
    assert result.status == "needs_review", (
        "Low-confidence retrieval must route to 'needs_review', not produce a guess"
    )


def test_low_confidence_result_explains_reason(monkeypatch, loaded_corpus):
    """AT-3: needs_review from confidence gate must explain retrieval/confidence reason."""
    from src.ingestion.chunker import Chunk
    from src.retrieval.search import ScoredChunk

    dummy = ScoredChunk(
        chunk=Chunk(
            doc_id="dummy", framework="x", version="v0",
            section_id="s0", text="irrelevant text",
        ),
        score=0.01,
    )
    monkeypatch.setattr("src.retrieval.search.dense_search", lambda q, k: [dummy])
    monkeypatch.setattr("src.retrieval.search.keyword_search", lambda q, k: [])

    result = assess_transaction(_PARSEABLE_TX)
    combined = " ".join(result.required_actions).lower()
    reason_keywords = {
        "retrieval", "confidence", "insufficient", "no relevant",
        "cannot determine", "not found",
    }
    assert any(kw in combined for kw in reason_keywords), (
        f"Low-confidence needs_review must explain the reason;\n"
        f"required_actions={result.required_actions!r}"
    )


# ── AT-4 — output always schema-valid; malformed state → needs_review ─────────

def test_output_is_always_a_compliance_assessment(complete_assessment):
    """AT-4: assess_transaction must return a ComplianceAssessment, never raw text."""
    assert isinstance(complete_assessment, ComplianceAssessment), (
        f"Expected ComplianceAssessment; got {type(complete_assessment)}"
    )


def test_validate_output_falls_back_on_malformed_state():
    """AT-4: validate_output falls back to needs_review when state has invalid schema."""
    bad_state: AgentState = {
        "raw_input": _COMPLETE_TX,
        "status": "assessed",
        "risk_rating": "INVALID_RATING_NOT_IN_LITERAL",
        "applicable_regulations": ["Basel III"],
        "required_actions": ["file report"],
        "citations": [],
    }
    result = validate_output(bad_state)
    assert result.get("status") == "needs_review", (
        f"validate_output must fall back to needs_review on invalid state; "
        f"got status={result.get('status')!r}"
    )


# ── AT-5 — citations in assessed results resolve to real ingested chunks ──────

def test_assessed_citations_resolve_to_ingested_chunks(
    complete_assessment, all_ingested_keys
):
    """AT-5: every citation must point to a chunk that was actually ingested."""
    if complete_assessment.status != "assessed":
        pytest.fail(
            f"AT-5 requires an 'assessed' result; got {complete_assessment.status!r}. "
            "Implement the agent so the $2M transaction returns 'assessed'."
        )

    for cit in complete_assessment.citations:
        assert (cit.doc_id, cit.section_id) in all_ingested_keys, (
            f"Citation ({cit.doc_id!r}, {cit.section_id!r}) does not resolve to any "
            "ingested chunk — possible hallucinated source reference."
        )


# ── AT-6 — final state carries a full audit trail ────────────────────────────

def test_state_contains_retrieved_regulations(complete_state):
    """AT-6: final state must hold the retrieved regulations."""
    assert complete_state.get("retrieved_regulations"), (
        "Audit trail: 'retrieved_regulations' absent or empty in final state"
    )


def test_state_contains_parsed_transaction(complete_state):
    """AT-6: final state must hold the parsed transaction fields."""
    assert complete_state.get("parsed_transaction"), (
        "Audit trail: 'parsed_transaction' absent or empty in final state"
    )


def test_state_contains_cross_reference_findings(complete_state):
    """AT-6: final state must hold cross-reference findings."""
    assert "cross_reference_findings" in complete_state, (
        "Audit trail: 'cross_reference_findings' key absent from final state"
    )
    assert isinstance(complete_state["cross_reference_findings"], list), (
        f"cross_reference_findings must be a list; "
        f"got {type(complete_state['cross_reference_findings'])}"
    )
