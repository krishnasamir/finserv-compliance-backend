"""Output schema for the compliance checker agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class RegCitation(BaseModel):
    """Source reference for a claim in a ComplianceAssessment."""

    doc_id: str
    section_id: str
    version: str


class ComplianceAssessment(BaseModel):
    """Structured compliance assessment returned by the agent.

    Pydantic validates this on every return — invalid LLM output never escapes
    the validate_output node as a raw string.
    """

    risk_rating: Literal["low", "medium", "high", "critical"] = "low"
    applicable_regulations: list[str] = []
    required_actions: list[str] = []
    citations: list[RegCitation] = []
    status: Literal["assessed", "needs_review"] = "needs_review"
