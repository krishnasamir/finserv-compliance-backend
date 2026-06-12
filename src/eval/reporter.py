"""Write eval_report.md with metric averages, per-question summary, and failure analysis."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.eval.runner import EvalResult


def report(
    metrics: dict,
    results: list[EvalResult],
    output_path: Path | None = None,
) -> None:
    """Write eval_report.md: metric averages, per-question answer table, failure analysis.

    output_path defaults to settings.eval_report_path when None.
    """
    from config import settings

    path = output_path or settings.eval_report_path

    lines: list[str] = [
        "# Evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Questions evaluated:** {len(results)}  ",
        f"**Judge model:** {settings.eval_judge_model} @ {settings.ollama_base_url}  ",
        "",
        "---",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Average |",
        "|---|---|",
    ]

    _display = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }
    for key in ("faithfulness", "answer_relevance", "context_precision", "context_recall"):
        if key in metrics:
            lines.append(f"| {_display[key]} | {metrics[key]:.3f} |")

    lines += [
        "",
        "---",
        "",
        "## Per-Question Answers",
        "",
        "| # | Question (truncated) | Answer (truncated) | Context chunks | Citations |",
        "|---|---|---|---|---|",
    ]

    for i, r in enumerate(results, 1):
        q = (r.question[:70] + "…") if len(r.question) > 70 else r.question
        a = (r.answer[:80] + "…") if len(r.answer) > 80 else r.answer
        q = q.replace("|", "&#124;")
        a = a.replace("|", "&#124;")
        lines.append(
            f"| {i} | {q} | {a} | {len(r.contexts)} | {len(r.citations)} |"
        )

    # ── Failure analysis ──────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Failure Analysis",
        "",
        "Questions that may need attention, with a one-line hypothesis:",
        "",
    ]

    failures: list[str] = []
    for r in results:
        if "insufficient context" in r.answer.lower():
            failures.append(
                f"- **{r.question[:80]}**  \n"
                f"  → System declined (out-of-corpus or below confidence threshold). "
                f"{'Expected — no sources in corpus.' if not r.expected_sources else 'Check whether the relevant document has been ingested.'}"
            )
        elif not r.citations:
            failures.append(
                f"- **{r.question[:80]}**  \n"
                f"  → Answer produced without citations — "
                "citation fallback may have failed or LLM refused to cite."
            )
        elif r.expected_sources and not any(
            c["doc_id"] in r.expected_sources for c in r.citations
        ):
            failures.append(
                f"- **{r.question[:80]}**  \n"
                f"  → Citations ({[c['doc_id'] for c in r.citations]}) "
                f"do not include expected sources ({r.expected_sources}). "
                "Retrieval may be surfacing the wrong document."
            )

    if failures:
        lines.extend(failures)
    else:
        lines.append(
            "No obvious failures detected — all questions produced cited answers "
            "from the expected sources."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
