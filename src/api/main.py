"""FastAPI application — exposes the RAG pipeline and compliance agent as HTTP endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.exceptions import EmptyQueryError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure transaction_log table exists on startup."""
    try:
        from src.storage.transaction_store import create_table
        create_table()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Could not create transaction_log table: %s", exc)
    yield


app = FastAPI(
    title="FinServ Compliance Assistant",
    description="RAG + agentic compliance checker over Basel III, MiFID II, and RBI regulatory text.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        return v


class CitationOut(BaseModel):
    doc_id: str
    section_id: str
    version: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]


class AssessRequest(BaseModel):
    scenario: str

    @field_validator("scenario")
    @classmethod
    def scenario_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("scenario must not be empty")
        return v


class AssessResponse(BaseModel):
    status: str
    risk_rating: str
    applicable_regulations: list[str]
    required_actions: list[str]
    citations: list[CitationOut]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline and return a cited answer.

    Data sovereignty: all inference runs locally via Ollama — no text is sent to
    an external API.
    """
    from src.rag.answer import answer

    try:
        ans = answer(req.question)
    except EmptyQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    return QueryResponse(
        answer=ans.text,
        citations=[
            CitationOut(doc_id=c.doc_id, section_id=c.section_id, version=c.version)
            for c in ans.citations
        ],
    )


class ChangeImpactRequest(BaseModel):
    doc_id: str

    @field_validator("doc_id")
    @classmethod
    def doc_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("doc_id must not be empty")
        return v


class AffectedSection(BaseModel):
    doc_id: str
    section_id: str
    framework: str
    version: str
    similarity_score: float


class ChangeImpactResponse(BaseModel):
    new_doc_id: str
    affected_sections: list[AffectedSection]
    affected_transaction_types: list[str]
    summary: str
    generated_at: str


class AuditReportRequest(BaseModel):
    transactions: list[str] | None = None  # if omitted, pulled from DB by date range
    period_start: str
    period_end: str

    @field_validator("transactions")
    @classmethod
    def transactions_not_empty_strings(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [t for t in v if t and t.strip()]


class AuditReportResponse(BaseModel):
    period_start: str
    period_end: str
    total_transactions: int
    risk_summary: dict[str, int]
    high_risk_transactions: list[dict]
    regulations_triggered: list[str]
    generated_at: str
    markdown: str


@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest) -> AssessResponse:
    """Run the LangGraph compliance-checker agent and return a structured assessment.

    Data sovereignty: all inference runs locally via Ollama.
    """
    from src.agent.graph import assess_transaction

    try:
        result = assess_transaction(req.scenario)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    # Redact PII before storing in audit log
    from src.security.pii_redactor import redact
    safe_scenario, pii_found = redact(req.scenario)
    if pii_found:
        import logging
        logging.getLogger(__name__).warning(
            "PII detected and redacted in scenario before storage: %s", pii_found
        )

    # Persist to transaction log for audit trail
    try:
        from src.storage.transaction_store import log_transaction
        log_transaction(
            scenario=safe_scenario,
            status=result.status,
            risk_rating=result.risk_rating,
            applicable_regulations=list(result.applicable_regulations),
            required_actions=list(result.required_actions),
            citations=[
                {"doc_id": c.doc_id, "section_id": c.section_id, "version": c.version}
                for c in result.citations
            ],
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Failed to log transaction: %s", exc)

    return AssessResponse(
        status=result.status,
        risk_rating=result.risk_rating,
        applicable_regulations=result.applicable_regulations,
        required_actions=result.required_actions,
        citations=[
            CitationOut(doc_id=c.doc_id, section_id=c.section_id, version=c.version)
            for c in result.citations
        ],
    )


@app.post("/analyze/change-impact", response_model=ChangeImpactResponse)
def change_impact(req: ChangeImpactRequest) -> ChangeImpactResponse:
    """Identify existing compliance sections affected by a newly ingested document.

    Uses hybrid similarity search — no external API calls.
    """
    from src.analysis.change_impact import analyze_change_impact

    try:
        report = analyze_change_impact(req.doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Change impact error: {exc}") from exc

    return ChangeImpactResponse(
        new_doc_id=report.new_doc_id,
        affected_sections=[AffectedSection(**s) for s in report.affected_sections],
        affected_transaction_types=report.affected_transaction_types,
        summary=report.summary,
        generated_at=report.generated_at,
    )


@app.post("/reports/audit", response_model=AuditReportResponse)
def audit_report(req: AuditReportRequest) -> AuditReportResponse:
    """Generate a structured compliance report for a set of transactions.

    If `transactions` is omitted, pulls all assessed transactions from the DB
    within the given date range. If provided, assesses them live via the agent.
    All inference is local via Ollama.
    """
    from src.reporting.audit_report import generate_audit_report
    from src.storage.transaction_store import get_transactions_by_period

    try:
        if req.transactions is None:
            # Pull from DB — report on already-assessed transactions
            stored = get_transactions_by_period(req.period_start, req.period_end)
            report = generate_audit_report(
                stored, req.period_start, req.period_end, from_db=True
            )
        else:
            # Assess the provided transactions live
            report = generate_audit_report(req.transactions, req.period_start, req.period_end)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit report error: {exc}") from exc

    return AuditReportResponse(
        period_start=report.period_start,
        period_end=report.period_end,
        total_transactions=report.total_transactions,
        risk_summary=report.risk_summary,
        high_risk_transactions=report.high_risk_transactions,
        regulations_triggered=report.regulations_triggered,
        generated_at=report.generated_at,
        markdown=report.markdown,
    )
