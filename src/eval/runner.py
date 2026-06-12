"""Run the 2A RAG pipeline over the eval set and collect (question, answer, contexts)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """One row of evaluation output — input/output/context bundle for scoring."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    expected_sources: list[str]
    citations: list[dict] = field(default_factory=list)


def run_eval(eval_set: list[dict]) -> list[EvalResult]:
    """Run answer_for_eval() on every question and return one EvalResult per question.

    Uses the 2A RAG pipeline (retrieve → rerank → compress → generate).
    Context texts are returned alongside the answer so DeepEval contextual
    metrics can run without a second round-trip to the DB.
    """
    from src.rag.answer import answer_for_eval

    results: list[EvalResult] = []
    for i, pair in enumerate(eval_set, 1):
        question = pair["question"]
        log.info("eval %d/%d: %s", i, len(eval_set), question[:60])
        try:
            ans, context_texts = answer_for_eval(question)
        except Exception as exc:
            log.error("RAG pipeline failed for question %r: %s", question[:60], exc)
            ans_text = f"Error: {exc}"
            context_texts = []
            citations: list[dict] = []
        else:
            ans_text = ans.text
            citations = [
                {"doc_id": c.doc_id, "section_id": c.section_id, "version": c.version}
                for c in ans.citations
            ]

        results.append(
            EvalResult(
                question=question,
                answer=ans_text,
                contexts=context_texts,
                ground_truth=pair["ground_truth"],
                expected_sources=pair.get("expected_sources", []),
                citations=citations,
            )
        )

    return results
