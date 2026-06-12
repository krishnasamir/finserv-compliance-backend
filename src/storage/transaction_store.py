"""Persistent transaction log — stores every /assess call for audit trail and reporting."""

from __future__ import annotations

import json
import logging

from sqlalchemy import text

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS transaction_log (
    id               SERIAL PRIMARY KEY,
    scenario         TEXT        NOT NULL,
    status           VARCHAR(20) NOT NULL,
    risk_rating      VARCHAR(20) NOT NULL,
    applicable_regulations JSONB NOT NULL DEFAULT '[]',
    required_actions       JSONB NOT NULL DEFAULT '[]',
    citations              JSONB NOT NULL DEFAULT '[]',
    assessed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_transaction_log_assessed_at ON transaction_log (assessed_at);
"""


def _engine():
    from src.ingestion.loader import _get_engine
    return _get_engine()


def create_table() -> None:
    """Create transaction_log table if it does not exist (idempotent)."""
    with _engine().begin() as conn:
        conn.execute(text(_DDL))


def log_transaction(
    scenario: str,
    status: str,
    risk_rating: str,
    applicable_regulations: list[str],
    required_actions: list[str],
    citations: list[dict],
) -> int:
    """Insert one assessment record and return its generated id."""
    sql = text("""
        INSERT INTO transaction_log
            (scenario, status, risk_rating, applicable_regulations, required_actions, citations)
        VALUES
            (:scenario, :status, :risk_rating, :applicable_regulations, :required_actions, :citations)
        RETURNING id
    """)
    with _engine().begin() as conn:
        row = conn.execute(sql, {
            "scenario": scenario,
            "status": status,
            "risk_rating": risk_rating,
            "applicable_regulations": json.dumps(applicable_regulations),
            "required_actions": json.dumps(required_actions),
            "citations": json.dumps(citations),
        }).fetchone()
    return int(row[0])


def get_transactions_by_period(period_start: str, period_end: str) -> list[dict]:
    """Return all logged transactions whose assessed_at falls within [period_start, period_end].

    Args:
        period_start: ISO date string (YYYY-MM-DD), inclusive.
        period_end:   ISO date string (YYYY-MM-DD), inclusive.

    Returns:
        List of dicts with keys: id, scenario, status, risk_rating,
        applicable_regulations, required_actions, citations, assessed_at.
    """
    sql = text("""
        SELECT id, scenario, status, risk_rating,
               applicable_regulations, required_actions, citations,
               assessed_at
        FROM transaction_log
        WHERE assessed_at::date BETWEEN :start AND :end
        ORDER BY assessed_at DESC
    """)
    with _engine().connect() as conn:
        rows = conn.execute(sql, {"start": period_start, "end": period_end}).fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row.id,
            "scenario": row.scenario,
            "status": row.status,
            "risk_rating": row.risk_rating,
            "applicable_regulations": row.applicable_regulations or [],
            "required_actions": row.required_actions or [],
            "citations": row.citations or [],
            "assessed_at": row.assessed_at.isoformat(),
        })
    return results
