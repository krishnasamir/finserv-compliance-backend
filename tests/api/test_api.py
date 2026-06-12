"""Acceptance tests for Phase 2D — FastAPI endpoints POST /query and POST /assess."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# AT-1  Health check
# ---------------------------------------------------------------------------

def test_health(client):
    """GET /health returns 200 and status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# AT-2  POST /query — valid question returns cited answer
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("loaded_corpus")
def test_query_returns_cited_answer(client):
    """/query returns text and a non-empty citations list for an in-corpus question."""
    resp = client.post("/query", json={"question": "What is KYC and what purpose does it serve?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 20
    assert "citations" in body
    assert isinstance(body["citations"], list)
    assert len(body["citations"]) > 0
    # Each citation must have doc_id and section_id
    for c in body["citations"]:
        assert "doc_id" in c
        assert "section_id" in c


# ---------------------------------------------------------------------------
# AT-3  POST /query — empty question returns 422
# ---------------------------------------------------------------------------

def test_query_empty_question_returns_422(client):
    """/query rejects an empty question with 422 Unprocessable Entity."""
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# AT-4  POST /query — missing field returns 422
# ---------------------------------------------------------------------------

def test_query_missing_field_returns_422(client):
    """/query rejects a body with no 'question' key with 422."""
    resp = client.post("/query", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# AT-5  POST /assess — valid scenario returns structured assessment
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("loaded_corpus")
def test_assess_returns_structured_output(client):
    """/assess returns a ComplianceAssessment with required fields."""
    resp = client.post("/assess", json={
        "scenario": "A bank onboards a new retail customer without collecting any identity documents."
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert body["status"] in ("assessed", "needs_review")
    assert "risk_rating" in body
    assert body["risk_rating"] in ("low", "medium", "high", "critical")
    assert "required_actions" in body
    assert isinstance(body["required_actions"], list)
    assert "citations" in body
    assert isinstance(body["citations"], list)


# ---------------------------------------------------------------------------
# AT-6  POST /assess — ambiguous input returns needs_review
# ---------------------------------------------------------------------------

def test_assess_ambiguous_returns_needs_review(client):
    """/assess returns needs_review for a vague, uncorroborated scenario."""
    resp = client.post("/assess", json={"scenario": "a payment was made"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "needs_review"


# ---------------------------------------------------------------------------
# AT-7  POST /assess — empty scenario returns 422
# ---------------------------------------------------------------------------

def test_assess_empty_scenario_returns_422(client):
    """/assess rejects an empty scenario with 422."""
    resp = client.post("/assess", json={"scenario": ""})
    assert resp.status_code == 422
