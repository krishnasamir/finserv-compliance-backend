"""LangGraph compliance checker graph — wires all nodes and gates."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_graph = None  # module-level cache so the graph is compiled once


def _build_graph():
    from langgraph.graph import END, StateGraph

    from src.agent.nodes import (
        assess_risk,
        check_confidence,
        check_sufficiency,
        classify_frameworks,
        cross_reference,
        flag_for_review,
        parse_input,
        retrieve,
        validate_output,
    )
    from src.agent.state import AgentState

    wf = StateGraph(AgentState)

    # Register nodes
    wf.add_node("parse_input", parse_input)
    wf.add_node("classify_frameworks", classify_frameworks)
    wf.add_node("retrieve", retrieve)
    wf.add_node("cross_reference", cross_reference)
    wf.add_node("assess_risk", assess_risk)
    wf.add_node("validate_output", validate_output)
    wf.add_node("flag_for_review", flag_for_review)

    # Entry point
    wf.set_entry_point("parse_input")

    # Gate 1 — sufficiency: are required transaction fields present?
    wf.add_conditional_edges(
        "parse_input",
        check_sufficiency,
        {"continue": "classify_frameworks", "flag_for_review": "flag_for_review"},
    )

    # Linear: classify → retrieve
    wf.add_edge("classify_frameworks", "retrieve")

    # Gate 2 — confidence: are retrieval scores above threshold?
    wf.add_conditional_edges(
        "retrieve",
        check_confidence,
        {"continue": "cross_reference", "flag_for_review": "flag_for_review"},
    )

    # Linear: cross_reference → assess_risk → validate_output → END
    wf.add_edge("cross_reference", "assess_risk")
    wf.add_edge("assess_risk", "validate_output")
    wf.add_edge("validate_output", END)

    # flag_for_review is terminal
    wf.add_edge("flag_for_review", END)

    return wf.compile()


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def _state_to_assessment(state: dict):
    """Convert final AgentState dict to a validated ComplianceAssessment."""
    from src.agent.schemas import ComplianceAssessment, RegCitation

    raw_citations = state.get("citations") or []
    reg_citations: list[RegCitation] = []
    for c in raw_citations:
        try:
            if isinstance(c, dict):
                reg_citations.append(RegCitation(**c))
        except Exception:
            pass

    try:
        return ComplianceAssessment(
            risk_rating=state.get("risk_rating", "low"),
            applicable_regulations=state.get("applicable_regulations") or [],
            required_actions=state.get("required_actions") or [],
            citations=reg_citations,
            status=state.get("status", "needs_review"),
        )
    except Exception as exc:
        log.error("_state_to_assessment failed: %s", exc)
        return ComplianceAssessment(
            status="needs_review",
            required_actions=["Assessment failed. Human review required."],
        )


def assess_transaction(description: str):
    """Run the compliance checker agent and return a structured ComplianceAssessment."""
    from src.agent.schemas import ComplianceAssessment

    assessment, _ = run_with_state(description)
    return assessment


def run_with_state(description: str):
    """Run the agent and return (ComplianceAssessment, final_state) for audit inspection."""
    from src.agent.state import AgentState

    graph = _get_graph()
    final_state: dict = graph.invoke({"raw_input": description})
    assessment = _state_to_assessment(final_state)
    return assessment, final_state
