"""Compliance audit report generation.

Generate a structured compliance report for a set of transactions over a period,
suitable for submission to an internal audit committee.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class AuditReport:
    """Structured compliance audit report for a set of transactions."""

    period_start: str
    period_end: str
    total_transactions: int
    risk_summary: dict[str, int]
    high_risk_transactions: list[dict]
    regulations_triggered: list[str]
    generated_at: str
    markdown: str = ""


def generate_audit_report(
    transactions: list[str] | list[dict],
    period_start: str,
    period_end: str,
    from_db: bool = False,
) -> AuditReport:
    """Assess each transaction and produce a structured audit report.

    Args:
        transactions: List of transaction scenario strings.
        period_start: Report period start date (YYYY-MM-DD).
        period_end:   Report period end date (YYYY-MM-DD).

    Returns:
        AuditReport with risk summary, high-risk list, regulations triggered,
        and a formatted markdown body ready for audit committee submission.
    """
    if not transactions:
        return AuditReport(
            period_start=period_start,
            period_end=period_end,
            total_transactions=0,
            risk_summary={},
            high_risk_transactions=[],
            regulations_triggered=[],
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            markdown=_render_markdown(period_start, period_end, 0, {}, [], []),
        )

    assessments = []

    if from_db:
        # Pre-assessed records from DB — use directly without calling the agent
        for rec in transactions:
            assessments.append({
                "transaction": rec["scenario"],
                "status": rec["status"],
                "risk_rating": rec["risk_rating"],
                "applicable_regulations": rec.get("applicable_regulations", []),
                "required_actions": rec.get("required_actions", []),
                "citations": rec.get("citations", []),
                "assessed_at": rec.get("assessed_at", ""),
            })
    else:
        from src.agent.graph import assess_transaction
        for txn in transactions:
            try:
                result = assess_transaction(txn)
                assessments.append({
                    "transaction": txn,
                    "status": result.status,
                    "risk_rating": result.risk_rating,
                    "applicable_regulations": list(result.applicable_regulations),
                    "required_actions": list(result.required_actions),
                    "citations": [
                        {"doc_id": c.doc_id, "section_id": c.section_id}
                        for c in result.citations
                    ],
                })
            except Exception as exc:
                log.warning("audit_report: assessment failed for %r: %s", txn[:60], exc)
                assessments.append({
                    "transaction": txn,
                    "status": "needs_review",
                    "risk_rating": "needs_review",
                    "applicable_regulations": [],
                    "required_actions": [f"Assessment error: {exc}"],
                    "citations": [],
                })

    # Risk summary: count by effective rating
    risk_counts: Counter = Counter()
    for a in assessments:
        key = a["risk_rating"] if a["status"] == "assessed" else "needs_review"
        risk_counts[key] += 1

    # High-risk transactions (high or critical)
    high_risk = [
        a for a in assessments
        if a["risk_rating"] in ("high", "critical") and a["status"] == "assessed"
    ]

    # Unique regulations triggered across all assessments
    all_regs: set[str] = set()
    for a in assessments:
        all_regs.update(a["applicable_regulations"])
    regulations_triggered = sorted(all_regs)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = _render_markdown(
        period_start, period_end, len(transactions),
        dict(risk_counts), high_risk, regulations_triggered,
        assessments, generated_at,
    )

    return AuditReport(
        period_start=period_start,
        period_end=period_end,
        total_transactions=len(transactions),
        risk_summary=dict(risk_counts),
        high_risk_transactions=high_risk,
        regulations_triggered=regulations_triggered,
        generated_at=generated_at,
        markdown=md,
    )


def _render_markdown(
    period_start: str,
    period_end: str,
    total: int,
    risk_summary: dict,
    high_risk: list[dict],
    regulations: list[str],
    assessments: list[dict] | None = None,
    generated_at: str = "",
) -> str:
    lines = [
        "# Compliance Audit Report",
        "",
        f"**Period:** {period_start} to {period_end}  ",
        f"**Generated:** {generated_at or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Total transactions assessed:** {total}  ",
        "",
        "---",
        "",
        "## Risk Summary",
        "",
        "| Risk Rating | Count |",
        "|---|---|",
    ]

    for rating in ("critical", "high", "medium", "low", "needs_review"):
        count = risk_summary.get(rating, 0)
        if count:
            lines.append(f"| {rating.capitalize()} | {count} |")

    lines += [
        "",
        "---",
        "",
        "## Regulations Triggered",
        "",
    ]
    if regulations:
        for reg in regulations:
            lines.append(f"- {reg}")
    else:
        lines.append("_None identified._")

    lines += [
        "",
        "---",
        "",
        "## High-Risk Transactions",
        "",
    ]
    if high_risk:
        for i, a in enumerate(high_risk, 1):
            lines += [
                f"### {i}. {a['transaction'][:100]}",
                f"**Risk:** {a['risk_rating']}  ",
                f"**Status:** {a['status']}  ",
                "**Required actions:**",
            ]
            for action in a["required_actions"]:
                lines.append(f"- {action}")
            if a["citations"]:
                lines.append("**Citations:**")
                for c in a["citations"]:
                    lines.append(f"- `{c['doc_id']}` / `{c['section_id']}`")
            lines.append("")
    else:
        lines.append("_No high-risk transactions identified._")

    lines += [
        "",
        "---",
        "",
        "## Full Transaction Log",
        "",
        "| # | Transaction | Status | Risk | Regulations |",
        "|---|---|---|---|---|",
    ]

    for i, a in enumerate(assessments or [], 1):
        txn = a["transaction"][:70] + "…" if len(a["transaction"]) > 70 else a["transaction"]
        regs = ", ".join(a["applicable_regulations"]) or "—"
        lines.append(f"| {i} | {txn} | {a['status']} | {a['risk_rating']} | {regs} |")

    lines += [
        "",
        "---",
        "",
        "*This report was generated automatically by the FinServ Compliance Assistant. "
        "All assessments are based on ingested regulatory text and must be reviewed "
        "by a qualified compliance officer before submission.*",
    ]

    return "\n".join(lines)
