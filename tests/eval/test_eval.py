"""AT-1 through AT-5: acceptance tests for Phase 2C — evaluation framework.

All tests are written BEFORE implementation (TDD).  With the current stubs
every test that checks implementation behaviour produces an AssertionError
(FAIL), not an ImportError / TypeError (ERROR).

Run order note: AT-2/3/4/5 depend on the DB being populated (they call the
2A pipeline internally) — they take `loaded_corpus` as a fixture so the
session-scoped corpus setup runs first.

Performance note: all Ollama calls are cached in session-scoped fixtures.
run_eval runs once (~11 calls), score runs once (~33 judge calls).
Total cold-start time is ~15-25 min on qwen2.5:3b; subsequent runs are fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import settings
from src.eval.reporter import report
from src.eval.runner import EvalResult, run_eval
from src.eval.scorer import score

# ── Shared helper ─────────────────────────────────────────────────────────────

def _load_eval_set() -> list[dict]:
    """Load data/eval_set.json from the configured path."""
    with open(settings.eval_set_path) as f:
        return json.load(f)


# ── Session-scoped fixtures — all Ollama calls happen exactly once ─────────────

@pytest.fixture(scope="session")
def all_results(loaded_corpus):
    """Run the full eval set through the pipeline once and cache the results."""
    return run_eval(_load_eval_set())


@pytest.fixture(scope="session")
def first3_results(loaded_corpus):
    """Run only the first 3 questions (fast subset for score/report tests)."""
    return run_eval(_load_eval_set()[:3])


@pytest.fixture(scope="session")
def first3_metrics(first3_results):
    """Score the first-3 results once and cache — used by AT-3 and AT-4."""
    return score(first3_results)


# ── AT-1 — eval_set.json is well-formed and contains 15–20 pairs ──────────────

def test_eval_set_json_exists_and_loadable():
    """AT-1a: eval_set.json must exist and parse as a non-empty JSON list."""
    assert settings.eval_set_path.exists(), (
        f"eval_set.json not found at {settings.eval_set_path}. "
        "Create the file with 15-20 hand-written question/ground_truth/expected_sources pairs."
    )
    data = _load_eval_set()
    assert isinstance(data, list) and len(data) > 0, "eval_set.json must be a non-empty list"


def test_eval_set_has_correct_count():
    """AT-1b: eval_set must have 15–20 pairs (scaffold has 3 — fill it in before this passes)."""
    data = _load_eval_set()
    assert 15 <= len(data) <= 20, (
        f"eval_set.json must contain 15–20 pairs; currently has {len(data)}. "
        "Add more pairs covering single-framework, cross-framework, and out-of-corpus questions."
    )


def test_eval_set_pairs_are_well_formed():
    """AT-1c: every pair must have question, ground_truth, expected_sources (list)."""
    data = _load_eval_set()
    required_keys = ("question", "ground_truth", "expected_sources")
    for i, pair in enumerate(data):
        for key in required_keys:
            assert key in pair, f"pair[{i}] ({pair.get('id', '?')!r}) is missing key {key!r}"
        assert isinstance(pair["expected_sources"], list), (
            f"pair[{i}].expected_sources must be a list; got {type(pair['expected_sources'])}"
        )
        assert pair["question"].strip(), f"pair[{i}].question must not be empty"
        assert pair["ground_truth"].strip(), f"pair[{i}].ground_truth must not be empty"


def test_eval_set_has_out_of_corpus_pairs():
    """AT-1d: eval_set must contain at least 2 out-of-corpus pairs (expected_sources=[])."""
    data = _load_eval_set()
    out_of_corpus = [p for p in data if p.get("expected_sources") == []]
    assert len(out_of_corpus) >= 2, (
        f"eval_set must have ≥ 2 out-of-corpus pairs (expected_sources=[]); "
        f"found {len(out_of_corpus)}. "
        "These validate that the system refuses gracefully when no relevant text exists."
    )


# ── AT-2 — run_eval produces one row per question ─────────────────────────────

def test_run_eval_returns_one_result_per_question(all_results):
    """AT-2a: run_eval must return exactly one EvalResult per input question."""
    data = _load_eval_set()
    assert len(all_results) == len(data), (
        f"run_eval must return one result per question; "
        f"got {len(all_results)} for {len(data)} questions."
    )


def test_run_eval_results_have_non_null_answers(all_results):
    """AT-2b: every EvalResult must have a non-empty answer and a contexts list."""
    assert len(all_results) > 0, "run_eval returned no results"
    for r in all_results:
        assert isinstance(r, EvalResult), f"Expected EvalResult; got {type(r)}"
        assert r.answer and r.answer.strip(), (
            f"EvalResult.answer must be non-empty for question: {r.question!r}"
        )
        assert r.contexts is not None, (
            f"EvalResult.contexts must not be None for question: {r.question!r}"
        )


def test_run_eval_results_carry_ground_truth(all_results):
    """AT-2c: EvalResult must echo back the ground_truth from the eval pair."""
    data = _load_eval_set()
    assert len(all_results) == len(data), "Pre-condition: must have one result per question"
    for pair, result in zip(data, all_results):
        assert result.ground_truth == pair["ground_truth"], (
            f"EvalResult.ground_truth must match the eval pair's ground_truth"
        )


# ── AT-3 — score returns all four metrics in [0,1]; judge is local ────────────

def test_score_returns_all_four_metric_keys(first3_metrics):
    """AT-3a: score() must return all four metric keys."""
    required = {"faithfulness", "answer_relevance", "context_precision", "context_recall"}
    assert isinstance(first3_metrics, dict), f"score() must return a dict; got {type(first3_metrics)}"
    missing = required - set(first3_metrics.keys())
    assert not missing, f"score() is missing metric keys: {missing}."


def test_score_metrics_are_floats_in_unit_interval(first3_metrics):
    """AT-3b: each metric value must be a float in [0, 1]."""
    for key in ("faithfulness", "answer_relevance", "context_precision", "context_recall"):
        assert key in first3_metrics, f"Missing metric key: {key}"
        val = first3_metrics[key]
        assert isinstance(val, float), f"{key} must be a float; got {type(val)}"
        assert 0.0 <= val <= 1.0, f"{key}={val} is outside [0,1]"


def test_score_judge_model_is_local_ollama():
    """AT-3c: the judge model must be configured to use the local Ollama endpoint.

    Config assertion — checks data-sovereignty constraint without calling the model.
    """
    assert settings.ollama_base_url.startswith("http://localhost"), (
        f"ollama_base_url must be a local endpoint; got {settings.ollama_base_url!r}."
    )
    external_indicators = ["api.openai.com", "api.anthropic.com", "openai.azure.com", "https://api."]
    assert not any(ext in settings.ollama_base_url for ext in external_indicators), (
        f"Judge model must NOT use an external API; got {settings.ollama_base_url!r}"
    )
    assert settings.eval_judge_model, "eval_judge_model must be set in config"


# ── AT-4 — report() writes eval_report.md with averages + failure analysis ────

def test_report_creates_eval_report_file(tmp_path, first3_results, first3_metrics):
    """AT-4a: report() must write a file to output_path."""
    report_path = tmp_path / "eval_report.md"
    report(first3_metrics, first3_results, output_path=report_path)
    assert report_path.exists(), (
        f"report() must write to {report_path}. Stub does nothing — implement report()."
    )


def test_report_contains_all_four_metric_averages(tmp_path, first3_results, first3_metrics):
    """AT-4b: eval_report.md must mention all four metric names."""
    report_path = tmp_path / "eval_report.md"
    report(first3_metrics, first3_results, output_path=report_path)
    content = report_path.read_text().lower()
    for metric in ("faithfulness", "answer relevance", "context precision", "context recall"):
        assert metric in content, f"eval_report.md must mention '{metric}'."


def test_report_contains_failure_analysis_section(tmp_path, first3_results, first3_metrics):
    """AT-4c: eval_report.md must contain a failure analysis section."""
    report_path = tmp_path / "eval_report.md"
    report(first3_metrics, first3_results, output_path=report_path)
    content = report_path.read_text().lower()
    assert "failure" in content or "analysis" in content, (
        "eval_report.md must contain a failure analysis section."
    )


# ── AT-5 — correct refusals for out-of-corpus questions score high faithfulness ─

def test_out_of_corpus_refusal_has_high_faithfulness(all_results):
    """AT-5: correctly refused out-of-corpus questions must score ≥ 0.7 faithfulness.

    Refusal is not hallucination — it must not be penalised.
    """
    out_of_corpus_results = [
        r for r in all_results if "insufficient context" in r.answer.lower()
    ]
    if not out_of_corpus_results:
        pytest.skip("No refusals detected in all_results — check out-of-corpus pairs.")

    metrics = score(out_of_corpus_results)
    assert "faithfulness" in metrics, "score() must return 'faithfulness'."
    assert metrics["faithfulness"] >= 0.7, (
        f"Correct refusals must score ≥ 0.7 faithfulness; got {metrics['faithfulness']:.3f}."
    )
