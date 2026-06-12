"""Acceptance tests for regulatory change impact analysis."""

from __future__ import annotations

import pytest


@pytest.mark.usefixtures("loaded_corpus")
def test_change_impact_returns_report():
    """Impact report is returned for a known ingested doc_id."""
    from src.analysis.change_impact import analyze_change_impact
    report = analyze_change_impact("KYC09062025")
    assert report is not None


@pytest.mark.usefixtures("loaded_corpus")
def test_change_impact_has_required_fields():
    """Report contains all required fields."""
    from src.analysis.change_impact import analyze_change_impact
    report = analyze_change_impact("KYC09062025")
    assert hasattr(report, "new_doc_id")
    assert hasattr(report, "affected_sections")
    assert hasattr(report, "affected_transaction_types")
    assert hasattr(report, "summary")
    assert hasattr(report, "generated_at")


@pytest.mark.usefixtures("loaded_corpus")
def test_change_impact_doc_id_matches():
    """Report doc_id matches the queried document."""
    from src.analysis.change_impact import analyze_change_impact
    report = analyze_change_impact("KYC09062025")
    assert report.new_doc_id == "KYC09062025"


@pytest.mark.usefixtures("loaded_corpus")
def test_change_impact_finds_affected_sections():
    """A KYC doc should affect sections in the other KYC document."""
    from src.analysis.change_impact import analyze_change_impact
    report = analyze_change_impact("KYC09062025")
    assert isinstance(report.affected_sections, list)
    assert len(report.affected_sections) > 0


@pytest.mark.usefixtures("loaded_corpus")
def test_change_impact_affected_sections_have_structure():
    """Each affected section has doc_id and section_id."""
    from src.analysis.change_impact import analyze_change_impact
    report = analyze_change_impact("KYC09062025")
    for section in report.affected_sections:
        assert "doc_id" in section
        assert "section_id" in section


@pytest.mark.usefixtures("loaded_corpus")
def test_change_impact_does_not_include_same_doc():
    """Affected sections must not be from the same document being analysed."""
    from src.analysis.change_impact import analyze_change_impact
    report = analyze_change_impact("KYC09062025")
    for section in report.affected_sections:
        assert section["doc_id"] != "KYC09062025"


@pytest.mark.usefixtures("loaded_corpus")
def test_change_impact_transaction_types_are_strings():
    """Affected transaction types must be a list of non-empty strings."""
    from src.analysis.change_impact import analyze_change_impact
    report = analyze_change_impact("KYC09062025")
    assert isinstance(report.affected_transaction_types, list)
    for t in report.affected_transaction_types:
        assert isinstance(t, str) and t.strip()


@pytest.mark.usefixtures("loaded_corpus")
def test_change_impact_unknown_doc_returns_empty():
    """Unknown doc_id returns a report with empty affected_sections."""
    from src.analysis.change_impact import analyze_change_impact
    report = analyze_change_impact("nonexistent-doc-xyz")
    assert report.affected_sections == []
