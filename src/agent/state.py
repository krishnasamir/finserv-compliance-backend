"""LangGraph state accumulated across compliance checker nodes."""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Mutable bag passed between nodes; all keys optional so nodes can add incrementally."""

    # Input
    raw_input: str

    # parse_input output
    parsed_transaction: dict[str, Any]
    missing_fields: list[str]

    # classify_frameworks output
    applicable_frameworks: list[str]

    # retrieve output
    retrieved_regulations: list[Any]   # list[ScoredChunk] — kept as Any to avoid circular import
    best_cosine_score: float
    best_keyword_score: float

    # cross_reference output
    cross_reference_findings: list[str]

    # assess_risk output
    risk_rating: str
    applicable_regulations: list[str]
    required_actions: list[str]
    citations: list[Any]               # list[dict] with doc_id/section_id/version keys

    # Terminal fields
    status: str
    error: Optional[str]
