# ADR-003: LangGraph for the Compliance Checker Agent

**Status:** Accepted  
**Date:** 2026-06-11  
**Deciders:** Architecture team  

---

## Context

The compliance checker needs to:
1. Parse a transaction description
2. Retrieve relevant regulations
3. Assess risk and produce a structured output
4. Handle ambiguous/insufficient input gracefully without fabricating

We evaluated three agent approaches:

| Option | Pros | Cons |
|---|---|---|
| **LangGraph** | Explicit graph, deterministic routing, auditable, typed state | More setup than simple chain |
| LangChain LCEL chain | Simpler, sequential | No conditional routing, hard to add gates, opaque failure paths |
| Custom Python (no framework) | Full control | No state management, more boilerplate, reinventing the wheel |

---

## Decision

Use **LangGraph** (`StateGraph`) to implement the compliance checker agent as an explicit directed graph with typed state, nodes, and conditional edges.

---

## Graph Structure

```
parse_input
    │
    ▼
check_sufficiency ──(flag)──► flag_for_review ──► END
    │ (continue)
    ▼
classify_frameworks
    │
    ▼
retrieve
    │
    ▼
check_confidence ──(flag)──► flag_for_review ──► END
    │ (continue)
    ▼
cross_reference
    │
    ▼
assess_risk
    │
    ▼
validate_output ──► END
```

---

## Alternatives Considered and Rationale

1. **Deterministic, auditable control flow** — every routing decision is an explicit Python function returning a string edge label. An auditor can trace exactly why a transaction was flagged vs assessed.
2. **First-class failure paths** — `flag_for_review` is a real node, not a try/except. Ambiguous input and low-confidence retrieval are designed paths, not error conditions.
3. **Two gates enforce graceful degradation:**
   - `check_sufficiency` — blocks on missing required transaction fields
   - `check_confidence` — blocks when retrieval scores are below threshold
   Both prevent the LLM from fabricating an assessment on insufficient evidence.
4. **Typed state** — `AgentState` (TypedDict) accumulates results across nodes. Every node reads and writes well-defined keys. No hidden side effects.
5. **Reuse of Phase 2A pipeline** — the `retrieve` node calls `hybrid_search` + `rerank` directly. No duplicate retrieval logic.
6. **Pydantic output validation** — `validate_output` node re-parses LLM output through `ComplianceAssessment`. Schema failure → `needs_review`. Invalid LLM output never escapes as a raw string.
7. **Open-source** — LangGraph is Apache 2.0 licensed.

---

## Consequences

- **Positive:** Control flow is inspectable — `run_with_state()` returns the full final state for audit inspection.
- **Positive:** Adding a new node (e.g., sanctions screening) requires adding one node function and one edge — no structural change.
- **Positive:** Each node is independently unit-testable via monkeypatching.
- **Negative:** More boilerplate than a simple sequential chain for straightforward cases.
- **Negative:** LangGraph's `StateGraph` compiles the graph at first call — adds ~1s cold start. Mitigated by module-level `_graph` cache.
