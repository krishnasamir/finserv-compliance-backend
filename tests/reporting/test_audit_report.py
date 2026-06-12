"""Acceptance tests for audit report generation."""

from __future__ import annotations

import pytest

_TRANSACTIONS = [
    "A walk-in customer requests a cash transaction of ₹75,000 at a bank branch.",
    "A bank onboards a new retail customer without collecting any identity documents.",
    "An international money transfer of $20,000 is requested by a walk-in customer to a beneficiary overseas.",
]


@pytest.mark.timeout(600)
@pytest.mark.usefixtures("loaded_corpus")
def test_audit_report_returns_report():
    """generate_audit_report returns a report object."""
    from src.reporting.audit_report import generate_audit_report
    report = generate_audit_report(_TRANSACTIONS, "2026-06-01", "2026-06-30")
    assert report is not None


@pytest.mark.usefixtures("loaded_corpus")
def test_audit_report_has_required_fields():
    """Report contains all required fields."""
    from src.reporting.audit_report import generate_audit_report
    report = generate_audit_report(_TRANSACTIONS, "2026-06-01", "2026-06-30")
    assert hasattr(report, "period_start")
    assert hasattr(report, "period_end")
    assert hasattr(report, "total_transactions")
    assert hasattr(report, "risk_summary")
    assert hasattr(report, "high_risk_transactions")
    assert hasattr(report, "regulations_triggered")
    assert hasattr(report, "generated_at")
    assert hasattr(report, "markdown")


@pytest.mark.usefixtures("loaded_corpus")
def test_audit_report_transaction_count():
    """total_transactions matches the number of inputs."""
    from src.reporting.audit_report import generate_audit_report
    report = generate_audit_report(_TRANSACTIONS, "2026-06-01", "2026-06-30")
    assert report.total_transactions == len(_TRANSACTIONS)


@pytest.mark.usefixtures("loaded_corpus")
def test_audit_report_risk_summary_keys():
    """risk_summary contains counts for valid risk ratings."""
    from src.reporting.audit_report import generate_audit_report
    report = generate_audit_report(_TRANSACTIONS, "2026-06-01", "2026-06-30")
    valid = {"low", "medium", "high", "critical", "needs_review"}
    for key in report.risk_summary:
        assert key in valid


@pytest.mark.usefixtures("loaded_corpus")
def test_audit_report_risk_summary_sums_to_total():
    """risk_summary counts sum to total_transactions."""
    from src.reporting.audit_report import generate_audit_report
    report = generate_audit_report(_TRANSACTIONS, "2026-06-01", "2026-06-30")
    assert sum(report.risk_summary.values()) == report.total_transactions


@pytest.mark.usefixtures("loaded_corpus")
def test_audit_report_high_risk_is_subset():
    """high_risk_transactions count does not exceed total."""
    from src.reporting.audit_report import generate_audit_report
    report = generate_audit_report(_TRANSACTIONS, "2026-06-01", "2026-06-30")
    assert len(report.high_risk_transactions) <= report.total_transactions


@pytest.mark.usefixtures("loaded_corpus")
def test_audit_report_regulations_triggered_are_strings():
    """regulations_triggered is a list of strings."""
    from src.reporting.audit_report import generate_audit_report
    report = generate_audit_report(_TRANSACTIONS, "2026-06-01", "2026-06-30")
    assert isinstance(report.regulations_triggered, list)
    for r in report.regulations_triggered:
        assert isinstance(r, str)


@pytest.mark.usefixtures("loaded_corpus")
def test_audit_report_markdown_contains_period():
    """Markdown report includes the period dates."""
    from src.reporting.audit_report import generate_audit_report
    report = generate_audit_report(_TRANSACTIONS, "2026-06-01", "2026-06-30")
    assert "2026-06-01" in report.markdown
    assert "2026-06-30" in report.markdown


def test_audit_report_empty_transactions():
    """Empty transaction list returns a report with zero count."""
    from src.reporting.audit_report import generate_audit_report
    report = generate_audit_report([], "2026-06-01", "2026-06-30")
    assert report.total_transactions == 0
    assert report.risk_summary == {}
