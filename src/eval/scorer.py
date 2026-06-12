"""Compute faithfulness, answer relevance, context precision, context recall via DeepEval."""

from __future__ import annotations

import logging
import os

# Extend per-attempt timeout and cap retries for local models on CPU.
# DEEPEVAL_RETRY_MAX_ATTEMPTS=1 means 1 retry (2 total attempts) then give up.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "1")

from src.eval.runner import EvalResult

log = logging.getLogger(__name__)

# Sentinel used when a question has no retrieved context (out-of-corpus).
# Keeps DeepEval from crashing on an empty list while making the semantics clear.
_EMPTY_CTX_SENTINEL = ["[no context retrieved — out-of-corpus or below confidence threshold]"]


def score(results: list[EvalResult]) -> dict:
    """Compute all four RAGAS-style metrics via DeepEval with the local Ollama judge.

    Returns a dict with keys:
        faithfulness, answer_relevance, context_precision, context_recall
    All values are floats in [0, 1].  Judge model is pointed at the local
    Ollama OpenAI-compatible endpoint per config (data sovereignty).
    """
    if not results:
        return {}

    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import LocalModel
    from deepeval.test_case import LLMTestCase

    from config import settings

    # Data sovereignty assertion — judge must NEVER call an external API.
    assert not any(
        ext in settings.ollama_base_url
        for ext in ["api.openai.com", "api.anthropic.com", "openai.azure.com"]
    ), f"Judge model must use local Ollama, not external API: {settings.ollama_base_url!r}"

    judge = LocalModel(
        model=settings.eval_judge_model,
        base_url=settings.ollama_base_url + "/v1",
        api_key="ollama",  # required by the OpenAI-compat client; value is ignored by Ollama
    )

    faith_m = FaithfulnessMetric(threshold=0.5, model=judge, async_mode=False)
    rel_m = AnswerRelevancyMetric(threshold=0.5, model=judge, async_mode=False)
    prec_m = ContextualPrecisionMetric(threshold=0.5, model=judge, async_mode=False)
    rec_m = ContextualRecallMetric(threshold=0.5, model=judge, async_mode=False)

    acc: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevance": [],
        "context_precision": [],
        "context_recall": [],
    }

    for r in results:
        is_refusal = "insufficient context" in r.answer.lower()
        ctx = r.contexts if r.contexts else _EMPTY_CTX_SENTINEL

        tc = LLMTestCase(
            input=r.question,
            actual_output=r.answer,
            expected_output=r.ground_truth,
            retrieval_context=ctx,
        )

        if not r.contexts and is_refusal:
            # A correct refusal with no context is maximally faithful — the
            # system made no claims unsupported by the (empty) retrieval.
            acc["faithfulness"].append(1.0)
            acc["context_precision"].append(1.0)
            acc["context_recall"].append(1.0)
        else:
            try:
                faith_m.measure(tc, _show_indicator=False)
                acc["faithfulness"].append(min(1.0, max(0.0, float(faith_m.score))))
            except Exception as exc:
                log.warning("faithfulness scoring failed for %r: %s", r.question[:50], exc)
                acc["faithfulness"].append(0.0)

            try:
                prec_m.measure(tc, _show_indicator=False)
                acc["context_precision"].append(min(1.0, max(0.0, float(prec_m.score))))
            except Exception as exc:
                log.warning("context_precision scoring failed for %r: %s", r.question[:50], exc)
                acc["context_precision"].append(0.0)

            try:
                rec_m.measure(tc, _show_indicator=False)
                acc["context_recall"].append(min(1.0, max(0.0, float(rec_m.score))))
            except Exception as exc:
                log.warning("context_recall scoring failed for %r: %s", r.question[:50], exc)
                acc["context_recall"].append(0.0)

        try:
            rel_m.measure(tc, _show_indicator=False)
            acc["answer_relevance"].append(min(1.0, max(0.0, float(rel_m.score))))
        except Exception as exc:
            log.warning("answer_relevance scoring failed for %r: %s", r.question[:50], exc)
            acc["answer_relevance"].append(0.0)

    return {k: sum(v) / len(v) for k, v in acc.items() if v}
