# Phase 2B — Agentic Compliance Checker (20 points)

A LangGraph agent that takes a transaction description, queries the 2A RAG pipeline, and returns a structured compliance assessment. Depends on Phase 2A being complete and green.

## Goal

Accept a transaction (e.g. "Cross-border payment of $2M to a non-KYC entity in a high-risk jurisdiction"), retrieve relevant regulations, and produce a structured, cited compliance assessment — handling ambiguous/insufficient input gracefully.

## State (typed)

`AgentState` (Pydantic/TypedDict) accumulates across nodes:
`raw_input, parsed_transaction, applicable_frameworks, retrieved_regulations, cross_reference_findings, risk_rating, required_actions, citations, status`.

## Graph nodes & edges

- `parse_input` — extract amount, counterparty, jurisdiction, instrument from free text.
- `check_sufficiency` (gate) — if required fields missing/ambiguous → route to `flag_for_review`.
- `classify_frameworks` — determine applicable frameworks (Basel III / MiFID II / RBI).
- `retrieve` — call 2A `hybrid_search` + rerank per framework.
- `check_confidence` (gate) — if top retrieval scores below threshold → route to `flag_for_review`.
- `cross_reference` — reconcile obligations across frameworks (strictest-wins).
- `assess_risk` — generate risk rating + required actions via local model, grounded in retrieved chunks.
- `validate_output` — enforce the Pydantic output schema; on failure, regenerate once.
- `flag_for_review` — terminal node returning a clear "insufficient information" result with what's missing.

## Output schema (Pydantic)

`ComplianceAssessment{ risk_rating: Literal["low","medium","high","critical"], applicable_regulations: list[str], required_actions: list[str], citations: list[Citation], status: Literal["assessed","needs_review"] }`

## Acceptance tests (write these first)

1. A complete transaction (the $2M example) returns `status="assessed"` with a valid risk_rating, ≥1 applicable regulation, ≥1 required action, and ≥1 resolvable citation.
2. An ambiguous input ("a payment was made") returns `status="needs_review"` and names the missing fields — never a fabricated rating.
3. A transaction whose retrieval scores are all below threshold routes to `needs_review`, not a low-confidence guess.
4. Output always validates against `ComplianceAssessment`; a malformed model response triggers exactly one regenerate, then `needs_review` if still invalid.
5. Every citation in an `assessed` result resolves to a real ingested chunk.
6. State is inspectable: after a run, the final state contains the retrieved regulations and reasoning used (audit trail).

## Definition of done

All acceptance tests green; the two gates (sufficiency, confidence) demonstrably route to review; output is always schema-valid; agent reuses the 2A pipeline (no duplicate retrieval logic).
