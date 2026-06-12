"""Regulatory change impact analysis.

When a new regulatory document is ingested, identify which existing
compliance sections and transaction types are affected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# Transaction types commonly affected by KYC/AML/capital regulations
_TRANSACTION_TYPE_KEYWORDS = {
    "KYC onboarding": ["onboard", "kyc", "customer identification", "cdd", "due diligence"],
    "Cash transactions": ["cash", "currency", "₹50,000", "50000"],
    "International transfers": ["international", "cross-border", "foreign", "overseas", "remittance"],
    "Demand Draft / Pay Order": ["demand draft", "pay order", "dd", "po"],
    "Small Accounts": ["small account", "simplified kyc"],
    "Video KYC (V-CIP)": ["v-cip", "video", "vcip"],
    "Periodic re-KYC": ["periodic", "re-kyc", "updation"],
    "Capital adequacy reporting": ["capital", "cet1", "tier 1", "rwa", "leverage"],
    "Investment products": ["investment", "derivative", "securities", "mifid"],
}


@dataclass
class ChangeImpactReport:
    """Structured report of regulatory change impact."""

    new_doc_id: str
    affected_sections: list[dict] = field(default_factory=list)
    affected_transaction_types: list[str] = field(default_factory=list)
    summary: str = ""
    generated_at: str = ""


def _get_doc_chunks(doc_id: str) -> list:
    """Fetch all stored chunks for a given doc_id from PGVector."""
    from sqlalchemy import text
    from src.ingestion.loader import _get_engine

    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT section_id, text FROM chunks WHERE doc_id = :doc_id LIMIT 20"),
            {"doc_id": doc_id},
        ).fetchall()
    return rows


def _infer_transaction_types(texts: list[str]) -> list[str]:
    """Infer affected transaction types from chunk texts using keyword matching."""
    combined = " ".join(texts).lower()
    affected = []
    for txn_type, keywords in _TRANSACTION_TYPE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            affected.append(txn_type)
    return affected


def analyze_change_impact(new_doc_id: str) -> ChangeImpactReport:
    """Identify existing compliance sections affected by a newly ingested document.

    Uses hybrid similarity search on the new document's chunks to find overlapping
    sections in other documents — no external API calls.

    Args:
        new_doc_id: doc_id of the newly ingested or updated regulatory document.

    Returns:
        ChangeImpactReport with affected sections (from other docs) and transaction types.
    """
    report = ChangeImpactReport(
        new_doc_id=new_doc_id,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    # Fetch the new document's chunks from the DB
    try:
        rows = _get_doc_chunks(new_doc_id)
    except Exception as exc:
        log.warning("change_impact: failed to fetch chunks for %r: %s", new_doc_id, exc)
        report.summary = f"Could not fetch chunks for {new_doc_id}."
        return report

    if not rows:
        report.summary = f"No chunks found for doc_id={new_doc_id!r}. Document may not be ingested."
        return report

    # Use dense search for real cosine similarity scores (RRF scores are not meaningful here)
    from src.retrieval.search import dense_search
    from config import settings

    # Only keep hits with cosine similarity above the retrieval threshold
    _SIMILARITY_THRESHOLD = settings.retrieval_score_threshold  # default 0.3

    seen: set[tuple[str, str]] = set()
    chunk_texts: list[str] = []

    for row in rows:
        section_id, text = row.section_id, row.text
        chunk_texts.append(text)

        try:
            hits = dense_search(text[:400], k=6)
        except Exception as exc:
            log.warning("change_impact: dense_search failed: %s", exc)
            continue

        for hit in hits:
            c = hit.chunk
            # Skip chunks from the same document and low-similarity hits
            if c.doc_id == new_doc_id:
                continue
            if hit.score < _SIMILARITY_THRESHOLD:
                continue
            key = (c.doc_id, c.section_id)
            if key in seen:
                continue
            seen.add(key)
            report.affected_sections.append({
                "doc_id": c.doc_id,
                "section_id": c.section_id,
                "framework": c.framework,
                "version": c.version,
                "similarity_score": round(hit.score, 3),
            })

    # Sort by similarity descending so highest-overlap sections appear first
    report.affected_sections.sort(key=lambda s: s["similarity_score"], reverse=True)

    # Infer affected transaction types from the new document's content
    report.affected_transaction_types = _infer_transaction_types(chunk_texts)

    # Build summary
    n_sections = len(report.affected_sections)
    n_txn = len(report.affected_transaction_types)
    affected_docs = sorted({s["doc_id"] for s in report.affected_sections})

    if n_sections == 0:
        report.summary = (
            f"No overlapping sections found in existing corpus for {new_doc_id}. "
            "This document may introduce entirely new regulatory coverage."
        )
    else:
        report.summary = (
            f"{new_doc_id} overlaps with {n_sections} section(s) across "
            f"{len(affected_docs)} existing document(s) "
            f"({', '.join(affected_docs)}). "
            f"{n_txn} transaction type(s) may be affected: "
            f"{', '.join(report.affected_transaction_types) or 'none identified'}."
        )

    return report
