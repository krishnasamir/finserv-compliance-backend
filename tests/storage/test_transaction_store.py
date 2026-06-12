"""Acceptance tests for transaction log storage."""

from __future__ import annotations

import pytest
from datetime import date


@pytest.fixture(autouse=True)
def ensure_table():
    from src.storage.transaction_store import create_table
    create_table()


def test_create_table_idempotent():
    """create_table can be called multiple times without error."""
    from src.storage.transaction_store import create_table
    create_table()
    create_table()


def test_log_and_retrieve_transaction():
    """A logged transaction is retrievable by date range."""
    from src.storage.transaction_store import log_transaction, get_transactions_by_period
    log_transaction(
        scenario="Test cash transaction of ₹75,000",
        status="assessed",
        risk_rating="high",
        applicable_regulations=["RBI-KYC"],
        required_actions=["Collect KYC documents"],
        citations=[{"doc_id": "KYC09062025", "section_id": "sec1", "version": "2025"}],
    )
    today = date.today().isoformat()
    results = get_transactions_by_period(today, today)
    assert len(results) >= 1
    assert any("75,000" in r["scenario"] for r in results)


def test_retrieved_record_has_required_fields():
    """Each retrieved record has all required fields."""
    from src.storage.transaction_store import log_transaction, get_transactions_by_period
    log_transaction(
        scenario="Field check transaction",
        status="needs_review",
        risk_rating="low",
        applicable_regulations=[],
        required_actions=["Human review required"],
        citations=[],
    )
    today = date.today().isoformat()
    results = get_transactions_by_period(today, today)
    assert results
    record = results[0]
    for field in ("id", "scenario", "status", "risk_rating",
                  "applicable_regulations", "required_actions", "citations", "assessed_at"):
        assert field in record, f"Missing field: {field}"


def test_date_range_filters_correctly():
    """Transactions outside the date range are not returned."""
    from src.storage.transaction_store import get_transactions_by_period
    results = get_transactions_by_period("2000-01-01", "2000-01-02")
    assert results == []


def test_log_transaction_returns_id():
    """log_transaction returns the new record's integer id."""
    from src.storage.transaction_store import log_transaction
    record_id = log_transaction(
        scenario="ID check transaction",
        status="assessed",
        risk_rating="medium",
        applicable_regulations=["Basel III"],
        required_actions=["Review capital ratio"],
        citations=[],
    )
    assert isinstance(record_id, int)
    assert record_id > 0
