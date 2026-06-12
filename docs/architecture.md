# Architecture Document — FinServ Compliance Assistant

**Version:** 2.0  
**Date:** 2026-06-11  
**Author:** Krishnasami R  

---

## 1. Overview

The FinServ Compliance Assistant is a Retrieval-Augmented Generation (RAG) + agentic compliance checker over Basel III, MiFID II, and RBI regulatory text. It answers natural-language compliance questions with cited sources and assesses transaction scenarios against ingested regulations.

**Hard constraints driving all design decisions:**
- **Data sovereignty** — no regulated text is sent to an external LLM API
- **Open-source only** — all models, libraries, and tools are open-source
- **Traceability** — every answer and assessment carries source citations
- **Graceful degradation** — never fabricates; returns `needs_review` on uncertainty

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT                                    │
│              (Swagger UI / curl / application)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│                     FastAPI  (src/api)                           │
│                                                                  │
│  GET  /health          POST /query        POST /assess           │
│  POST /analyze/        POST /reports/                            │
│       change-impact         audit                                │
└────────┬──────────────────────┬────────────────────────────────-┘
         │                      │
┌────────▼──────────┐  ┌────────▼──────────────────────────────┐
│  RAG Pipeline     │  │  LangGraph Agent  (src/agent)         │
│  (src/rag)        │  │                                        │
│                   │  │  parse_input                           │
│  hybrid_search    │  │  check_sufficiency ──► flag_for_review │
│  rerank           │  │  classify_frameworks                   │
│  compress         │  │  retrieve                              │
│  generate         │  │  check_confidence  ──► flag_for_review │
│                   │  │  cross_reference                       │
└────────┬──────────┘  │  assess_risk                           │
         │             │  validate_output                       │
         └──────┬───────┘                                        │
                │        └───────────────────────────────────────┘
┌───────────────▼──────────────────────────────────────────────┐
│                   src/retrieval                               │
│                                                               │
│   dense_search()     keyword_search()     hybrid RRF fusion  │
│   (PGVector cosine)  (tsvector BM25)      + BGE reranker     │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│              PostgreSQL + PGVector  (Docker)                  │
│                                                               │
│   chunks table:                                               │
│     doc_id | framework | version | section_id                │
│     text | embedding (vector 384) | tsv (tsvector)           │
│                                                               │
│   transaction_log table:                                      │
│     scenario | status | risk_rating | regulations            │
│     required_actions | citations | assessed_at               │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                   src/ingestion                               │
│                                                               │
│   parse_pdf()  →  chunk()  →  embed()  →  load()             │
│   PyMuPDF         512-token    BGE-small   PGVector upsert   │
│                   overlap 12%  en-v1.5                        │
└──────────────────────────────────────────────────────────────┘

            All LLM inference: Ollama (local, http://localhost:11434)
            Models: qwen2.5:3b (RAG/agent) | mistral:7b (eval judge)
```

---

## 3. Component Descriptions

### 3.1 Ingestion Pipeline (`src/ingestion/`)

| Module | Function | Technology |
|---|---|---|
| `parser.py` | Extract text per page, preserve section structure | PyMuPDF |
| `chunker.py` | Clause-aware chunking, ~512 tokens, 12% overlap | Custom splitter |
| `embedder.py` | Generate 384-dim dense vectors | `BAAI/bge-small-en-v1.5` |
| `loader.py` | Upsert chunks + tsvector into Postgres | SQLAlchemy + PGVector |

Each chunk carries metadata: `doc_id`, `framework`, `version`, `section_id`, `text`, `embedding`, `tsv`.

### 3.2 Retrieval (`src/retrieval/`)

Hybrid retrieval fuses two signals via Reciprocal Rank Fusion (RRF):

```
Query
  ├── dense_search()    →  top-10 by cosine similarity (PGVector)
  └── keyword_search()  →  top-10 by tsvector BM25 ranking
              │
              ▼
    reciprocal_rank_fusion()   →  deduplicated, re-ranked list
              │
              ▼
    rerank()   →  BGE cross-encoder reranker → top-4 chunks
```

**Why two-stage ranking?**
- Stage 1 (RRF): fast approximate ranking combining semantic + lexical signals
- Stage 2 (BGE reranker): slow but accurate cross-encoder that scores query-chunk pairs jointly — catches cases where the bi-encoder embedding missed relevance

**Confidence gate:** if best cosine score < 0.3 (config), the query is considered out-of-corpus and the LLM is never called.

### 3.3 RAG Pipeline (`src/rag/`)

```
question
    │
    ▼
hybrid_search + rerank
    │
    ▼
[cosine < threshold?] ── YES ──► "Insufficient context" (no fabrication)
    │ NO
    ▼
compress()   → extract most relevant sentences per chunk
    │
    ▼
Ollama (qwen2.5:3b)  → cited answer with [REF-N] markers
    │
    ▼
_parse_citations()   → map [REF-N] → {doc_id, section_id, version}
    │
    ▼
Answer(text, citations=[...])
```

### 3.4 Compliance Agent (`src/agent/`)

A LangGraph `StateGraph` with typed `AgentState`. See [ADR-003](adr/ADR-003-langgraph-agent-framework.md) for full rationale.

```
parse_input  →  check_sufficiency  →  classify_frameworks  →  retrieve
                      │                                           │
               flag_for_review                         check_confidence
               (needs_review)                                     │
                                                          flag_for_review
                                                          (needs_review)
                                                                  │
                                                         cross_reference
                                                                  │
                                                           assess_risk
                                                                  │
                                                         validate_output
                                                          (Pydantic check)
                                                                  │
                                                    ComplianceAssessment output
```

**Output schema (Pydantic):**
```python
class ComplianceAssessment(BaseModel):
    risk_rating: Literal["low", "medium", "high", "critical"]
    applicable_regulations: list[str]
    required_actions: list[str]
    citations: list[RegCitation]
    status: Literal["assessed", "needs_review"]
```

### 3.5 Evaluation Framework (`src/eval/`)

| Module | Responsibility |
|---|---|
| `runner.py` | Run RAG pipeline on eval set, collect (answer, context, ground_truth) |
| `scorer.py` | Score with DeepEval (4 RAGAS-style metrics), judge = local Ollama |
| `reporter.py` | Write `eval_report.md` with metric averages + failure analysis |

**4 Metrics:** Faithfulness, Answer Relevance, Context Precision, Context Recall — all scored by `mistral:7b` via local Ollama. Data sovereignty assertion enforced in code.

### 3.6 API (`src/api/`)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/query` | POST | RAG pipeline — cited answer |
| `/assess` | POST | Agent — structured compliance assessment + auto-logs to DB |
| `/analyze/change-impact` | POST | Identify sections affected by a new regulatory document |
| `/reports/audit` | POST | Compliance audit report for a date range (pulls from DB) |

### 3.7 Change Impact Analysis (`src/analysis/`)

When a new regulatory document is ingested, the system identifies which existing compliance sections and transaction types overlap with it using dense similarity search.

```
new doc_id
    │
    ▼
fetch all chunks for doc_id from DB (up to 20)
    │
    ▼
for each chunk → dense_search(chunk_text, k=6)
    │
    ▼
filter: remove same-doc hits + cosine < 0.3
    │
    ▼
deduplicate by (doc_id, section_id)
    │
    ▼
sort by similarity score (descending)
    │
    ▼
infer affected transaction types via keyword matching
    │
    ▼
ChangeImpactReport(affected_sections, transaction_types, summary)
```

### 3.8 Security (`src/security/`)

| Module | Responsibility |
|---|---|
| `pii_redactor.py` | Regex-based detection and redaction of Aadhaar, PAN, phone, email, bank account, IFSC before any data is persisted |

Called by FastAPI `/assess` before `log_transaction()`. Returns `(safe_text, list[pii_types_found])`.

### 3.9 Audit Report Generation (`src/reporting/`)

Generates a structured compliance report for an audit committee, either from live assessment of new transactions or from the historical `transaction_log` table.

```
POST /reports/audit
    │
    ├── transactions provided? ── YES ──► assess each via LangGraph agent
    │                                     log each to transaction_log
    │
    └── NO (date range only) ────────► query transaction_log by assessed_at
                │
                ▼
    aggregate: risk_summary (count by rating)
               high_risk_transactions (risk = high/critical)
               regulations_triggered (unique set)
                │
                ▼
    render markdown → AuditReport(risk_summary, high_risk, markdown)
```

---

## 4. Sequence Diagrams

### 4.1 Full RAG Query Sequence

```
Client          FastAPI         RAG             Retrieval       Postgres        Ollama
  │                │              │                 │               │              │
  │─POST /query───►│              │                 │               │              │
  │                │─answer(q)───►│                 │               │              │
  │                │              │─embed(q)────────┤               │              │
  │                │              │◄────vector───────┤               │              │
  │                │              │─dense_search(k=10)──────────────►│              │
  │                │              │◄────10 chunks────────────────────│              │
  │                │              │─keyword_search(k=10)────────────►│              │
  │                │              │◄────10 chunks────────────────────│              │
  │                │              │─RRF fusion + rerank──────────────┤              │
  │                │              │◄────top 4 chunks─────────────────┤              │
  │                │              │                 │               │              │
  │                │              │ [cosine < 0.3?]──►return "Insufficient context" │
  │                │              │                 │               │              │
  │                │              │─compress(chunks)─┤               │              │
  │                │              │─ollama.chat(prompt+context)──────────────────►  │
  │                │              │◄────answer text with [REF-N]─────────────────── │
  │                │              │─parse_citations()┤               │              │
  │                │◄─Answer──────│                 │               │              │
  │◄─{answer,citations}───────────│                 │               │              │
```

### 4.2 Full Agent Assessment Sequence

```
Client      FastAPI      Agent            Retrieval      Postgres      Ollama
  │            │           │                  │              │            │
  │─POST /assess──────────►│                  │              │            │
  │            │─assess_transaction(scenario)►│              │            │
  │            │           │─parse_input──────────────────────────────►  │
  │            │           │◄──{amount,counterparty,jurisdiction}──────── │
  │            │           │                  │              │            │
  │            │           │ [missing≥2?]──►flag_for_review (needs_review)│
  │            │           │                  │              │            │
  │            │           │─classify_frameworks (rule-based)│            │
  │            │           │─hybrid_search(scenario, k=10)──►│            │
  │            │           │◄──fused+reranked chunks──────────│            │
  │            │           │                  │              │            │
  │            │           │ [cosine<0.3?]──►flag_for_review (needs_review)
  │            │           │                  │              │            │
  │            │           │─cross_reference()│              │            │
  │            │           │─assess_risk──────────────────────────────►  │
  │            │           │◄──{risk_rating, required_actions}─────────── │
  │            │           │─validate_output (Pydantic check)│            │
  │            │◄──ComplianceAssessment───────│              │            │
  │            │                              │              │            │
  │            │─pii_redactor.redact(scenario)│              │            │
  │            │  → safe_scenario (Aadhaar/PAN/phone/email stripped)      │
  │            │─log_transaction(safe_scenario, status, risk)────────────►│
  │◄─{status,risk,citations}──────────────────│              │            │
```

### 4.3 Audit Report by Date Range Sequence

```
Client       FastAPI        Storage           Reporting
  │             │               │                 │
  │─POST /reports/audit────────►│                 │
  │  {period_start, period_end} │                 │
  │             │─get_transactions_by_period()────►│
  │             │◄──list[transaction records]──────│
  │             │─generate_audit_report(records)───────────────►│
  │             │◄──AuditReport(risk_summary, markdown)─────────│
  │◄─{risk_summary, markdown}───│                 │
```

---

## 5. Error Handling Paths

Every failure in the system is a handled path — not a crash. The table below shows what happens at each failure point.

### 5.1 RAG Pipeline Errors

| Failure Point | Error | Handling | Response |
|---|---|---|---|
| Empty question | `EmptyQueryError` | Raised before any DB call | HTTP 422 |
| No chunks retrieved | Empty list | `best_cosine = 0.0 < 0.3` | `"Insufficient context"` answer, no citations |
| Cosine below threshold | Score < 0.3 | Retrieval gate closes | `"Insufficient context"` answer |
| Ollama timeout | `OllamaTimeoutError` | Caught in `_pipeline()` | HTTP 500 with detail |
| LLM returns no `[REF-N]` | Empty citation parse | `citations = []` | Answer returned, empty citations |
| Postgres down | `OperationalError` | Propagates to FastAPI | HTTP 500 with detail |

### 5.2 Agent Errors

| Failure Point | Error | Handling | Response |
|---|---|---|---|
| Empty scenario | `ValueError` | Pydantic validator | HTTP 422 |
| ≥2 fields missing | `missing_fields` set | `check_sufficiency` gate | `needs_review` + field list |
| Retrieval below threshold | Low cosine | `check_confidence` gate | `needs_review` + confidence score |
| LLM returns invalid JSON | Parse fails | `_parse_llm_json()` best-effort, then `{}` | `needs_review` via `validate_output` |
| Pydantic validation fails | `ValidationError` | `validate_output` catches | `needs_review` + schema error note |
| Assessed but no citations | Business rule | `validate_output` rejects | `needs_review` |
| Agent node exception | `Exception` | `graph.py` outer try/except | `needs_review` fallback assessment |

### 5.3 Evaluation Errors

| Failure Point | Error | Handling | Response |
|---|---|---|---|
| Judge outputs invalid JSON | `ValidationError` | Caught per-metric | Score recorded as `0.0` |
| Judge timeout (>600s) | Timeout | `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE` | Score `0.0` after 2 attempts |
| Out-of-corpus question | Empty context | Shortcut: skip judge call | Faithfulness/Precision/Recall = `1.0` |

### 5.4 API-Level Errors

```
Request
    │
    ▼
Pydantic validation ──FAIL──► HTTP 422 (field errors, empty inputs)
    │ PASS
    ▼
Business logic ──────FAIL──► HTTP 500 (detail: error message)
    │ PASS
    ▼
HTTP 200 (structured response)
```

---

## 6. New Features

### 6.1 Regulatory Change Impact Analysis

**Problem:** When a compliance team ingests a new RBI circular, they need to know which existing policies and transaction types are affected — without reading every document manually.

**Solution:** `POST /analyze/change-impact` uses cosine similarity to find overlapping sections between the new document and all existing corpus documents.

```
Input:  {"doc_id": "KYC09062025"}

Step 1: Fetch all chunks of KYC09062025 from DB
Step 2: For each chunk → dense_search(chunk_text, k=6)
Step 3: Filter out: same-doc hits, cosine < 0.3 (retrieval_score_threshold)
Step 4: Deduplicate by (doc_id, section_id), sort by score DESC
Step 5: Keyword-match chunk texts → affected transaction types
Step 6: Return ChangeImpactReport

Output:
{
  "new_doc_id": "KYC09062025",
  "affected_sections": [
    {"doc_id": "MD18KYC...", "section_id": "RBI/DBR-c49",
     "framework": "RBI-KYC", "similarity_score": 0.72}
  ],
  "affected_transaction_types": ["KYC onboarding", "Cash transactions", "V-CIP"],
  "summary": "KYC09062025 overlaps with 8 sections across 1 document..."
}
```

**Design decision:** Uses `dense_search` (cosine), not `hybrid_search` (RRF). RRF scores are always ~0.016 and carry no similarity meaning. Cosine scores are in [0,1] and directly comparable.

### 6.2 Audit Report Generation with Transaction Log

**Problem:** Compliance teams need periodic reports (monthly/quarterly) showing all assessed transactions, risk distribution, and regulations triggered — suitable for internal audit committees.

**Solution:**
1. Every `POST /assess` call automatically writes to `transaction_log` (Postgres)
2. `POST /reports/audit` reads the log by date range and generates a structured markdown report

**Two modes:**
```
Mode 1 — Historical (date range only):
POST /reports/audit
{"period_start": "2026-06-01", "period_end": "2026-06-30"}
→ Reads transaction_log, no new LLM calls needed

Mode 2 — Fresh batch assessment:
POST /reports/audit
{"transactions": ["...", "..."], "period_start": "...", "period_end": "..."}
→ Runs each through the agent, logs them, then reports
```

**Report contents:**
- Risk summary table (count by rating: critical/high/medium/low/needs_review)
- Unique regulations triggered across all transactions
- Full detail for high-risk transactions (actions + citations)
- Complete transaction log table
- Markdown formatted for audit committee submission

---

## 7. Deployment View

```
┌─────────────────────────── Host Machine (WSL2 / Linux) ──────────────────────┐
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        Docker Compose                                   │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────────────────┐                               │  │
│  │  │  finserv_pgvector                    │                               │  │
│  │  │  image: pgvector/pgvector:pg16       │                               │  │
│  │  │  port:  5432                         │                               │  │
│  │  │  volume: pgdata (persistent)         │                               │  │
│  │  │  healthcheck: pg_isready             │                               │  │
│  │  └──────────────────────────────────────┘                               │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │  Python venv (.venv)                                                  │    │
│  │                                                                       │    │
│  │  uvicorn src.api.main:app --port 8000                                 │    │
│  │    └── FastAPI app                                                    │    │
│  │         ├── src/rag/         (BGE-small loaded at first request)      │    │
│  │         ├── src/retrieval/   (BGE-reranker loaded at first request)   │    │
│  │         ├── src/agent/       (LangGraph compiled at first call)       │    │
│  │         ├── src/analysis/                                             │    │
│  │         ├── src/reporting/                                            │    │
│  │         └── src/storage/                                              │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │  Ollama (system service)                                              │    │
│  │  port: 11434                                                          │    │
│  │  models: qwen2.5:3b (1.9 GB) | mistral:7b (4.4 GB)                   │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

Ports exposed:
  8000  →  FastAPI (Swagger UI at http://localhost:8000/docs)
  5432  →  PostgreSQL (internal only, not exposed to host)
  11434 →  Ollama (internal only)

Startup order:
  1. docker compose up -d          (Postgres)
  2. ollama serve                  (auto-started by Ollama install)
  3. python -m src.ingestion.run   (one-time: ingest PDFs)
  4. uvicorn src.api.main:app      (API server)
```

**Configuration:** All environment-specific values live in `.env` (never committed):
```
DATABASE_URL=postgresql://finserv:finserv_local_dev@localhost:5432/compliance
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:3b
EVAL_JUDGE_MODEL=mistral:7b
```
Production changes only `LLM_MODEL` and `DATABASE_URL`. No code changes required.

---

## 8. Security Considerations

### 8.1 Data Sovereignty (enforced in code)
All LLM inference routes to `OLLAMA_BASE_URL` (default: `http://localhost:11434`). A runtime assertion in `scorer.py` prevents the judge model from accidentally pointing at an external API:
```python
assert not any(ext in settings.ollama_base_url
               for ext in ["api.openai.com", "api.anthropic.com", "openai.azure.com"])
```
Regulated text (RBI KYC, Basel III) never leaves the host machine.

### 8.2 Secrets Management
- `.env` is listed in `.gitignore` — never committed
- `.env.example` (committed) shows required keys without values
- No API keys required at runtime (Ollama needs none; `api_key="ollama"` in DeepEval is a placeholder)

### 8.3 Input Validation
All API inputs are validated by Pydantic before reaching business logic:
- Empty strings rejected with HTTP 422
- Unknown fields silently ignored (`extra="ignore"` in Settings)
- No raw SQL constructed from user input — all queries use SQLAlchemy `text()` with bound parameters (parameterised queries prevent SQL injection)

### 8.4 PII Redaction Before Storage
`src/security/pii_redactor.py` runs on every `/assess` call before the scenario is written to `transaction_log`. Detected patterns are replaced with labelled placeholders:

| Pattern | Regex target | Replacement |
|---|---|---|
| Aadhaar | 12-digit groups | `[AADHAAR_REDACTED]` |
| PAN | `ABCDE1234F` format | `[PAN_REDACTED]` |
| Phone | +91 / 6-9XXXXXXXXX | `[PHONE_REDACTED]` |
| Email | standard email | `[EMAIL_REDACTED]` |
| Bank account | 9–18 digit sequences | `[ACCOUNT_REDACTED]` |
| IFSC code | `AAAA0XXXXXX` format | `[IFSC_REDACTED]` |

The original (un-redacted) scenario is used for assessment but never persisted. A `WARNING` log entry records which PII types were found.

- Logging is at `INFO`/`WARNING` level — chunk texts and LLM outputs are not logged at `DEBUG` by default

### 8.5 Limitations (prototype scope)
The following production-grade controls are **not implemented** in this prototype:
- No authentication / authorisation on API endpoints
- No TLS (HTTP only)
- No rate limiting or per-user quotas
- No audit log for who called which endpoint

---

## 9. Database Schema

```sql
-- Regulatory text chunks
CREATE TABLE chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    framework   TEXT NOT NULL,
    version     TEXT NOT NULL,
    section_id  TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   VECTOR(384),          -- BGE-small cosine index
    tsv         TSVECTOR,             -- keyword index
    UNIQUE (doc_id, section_id)
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING GIN (tsv);

-- Compliance assessment audit log
CREATE TABLE transaction_log (
    id                     SERIAL PRIMARY KEY,
    scenario               TEXT        NOT NULL,
    status                 VARCHAR(20) NOT NULL,
    risk_rating            VARCHAR(20) NOT NULL,
    applicable_regulations JSONB       NOT NULL DEFAULT '[]',
    required_actions       JSONB       NOT NULL DEFAULT '[]',
    citations              JSONB       NOT NULL DEFAULT '[]',
    assessed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ON transaction_log (assessed_at);
```

---

## 10. Technology Stack

| Layer | Technology | Licence |
|---|---|---|
| Language | Python 3.11 | PSF |
| LLM inference | Ollama + qwen2.5:3b / mistral:7b | MIT / Apache 2.0 |
| Embeddings | BAAI/bge-small-en-v1.5 | MIT |
| Reranker | BAAI/bge-reranker-base | MIT |
| Vector store | PostgreSQL 16 + PGVector | PostgreSQL / MIT |
| Agent framework | LangGraph | MIT |
| API framework | FastAPI + Uvicorn | MIT |
| Validation | Pydantic v2 | MIT |
| Evaluation | DeepEval | Apache 2.0 |
| PDF parsing | PyMuPDF | AGPL / commercial |
| Tests | pytest | MIT |
| Containerisation | Docker Compose | Apache 2.0 |

---

## 11. Architecture Decision Records

- [ADR-001: PGVector as the vector store](adr/ADR-001-pgvector-vector-store.md)
- [ADR-002: Local Ollama for all LLM inference](adr/ADR-002-local-ollama-inference.md)
- [ADR-003: LangGraph for the compliance agent](adr/ADR-003-langgraph-agent-framework.md)

---

## 12. Multi-Model Routing Strategy

The system uses two models with explicit routing based on task complexity:

| Task | Model | Rationale |
|---|---|---|
| Transaction field extraction (`parse_input`) | `qwen2.5:3b` | Structured JSON extraction from short text — 3B sufficient |
| Framework classification (`classify_frameworks`) | Rule-based (no LLM) | Keyword matching is deterministic and faster than LLM |
| Risk assessment (`assess_risk`) | `qwen2.5:3b` | Grounded generation from retrieved context |
| Answer generation (`/query`) | `qwen2.5:3b` | Retrieval-grounded — context window is the constraint, not model size |
| Evaluation judge (`scorer.py`) | `mistral:7b` | Structured verdict JSON requires better instruction following than 3B |

**Production routing upgrade path:**

```
                    ┌─── short classification tasks ──► qwen2.5:3b  (fast, cheap)
Query complexity ───┤
                    └─── complex multi-framework assessment ──► llama3:70b (accurate)
```

In production, a router node inspects query complexity (number of frameworks, transaction amount, jurisdiction risk level) and routes to the appropriate model. Model name is a config value — no code change required to swap.

---

## 13. Prompt Engineering Framework

All LLM prompts follow a consistent structure across the codebase:

### System Prompt Pattern
```
You are a regulatory compliance expert. Answer the question using ONLY the
regulatory text below. [ROLE CONSTRAINT]

REGULATORY CONTEXT:
{context}     ← retrieved chunks with [Source N: doc_id | framework | section]

QUESTION / TRANSACTION: {query}
```

### Prompt Templates by Node

**RAG answer generation** (`src/rag/answer.py`):
- Role: regulatory compliance expert
- Constraint: "use ONLY the regulatory text below"
- Output: free text with `[REF-N]` citation markers
- Design: zero-shot — regulatory text provides sufficient grounding

**Transaction parsing** (`src/agent/nodes.py — parse_input`):
- Role: field extractor
- Constraint: "ONLY a valid JSON object — no additional text"
- Output: `{amount, currency, counterparty, jurisdiction, instrument}`
- Design: structured output prompt with explicit null-handling rules

**Risk assessment** (`src/agent/nodes.py — assess_risk`):
- Role: regulatory compliance expert
- Constraint: JSON only, cite specific regulation names from context
- Output: `{risk_rating, applicable_regulations, required_actions}`
- Design: context-grounded with explicit rating scale (low/medium/high/critical)

### Chain-of-Thought Design
The agent implements implicit chain-of-thought via **node sequencing**:
```
parse → classify → retrieve → cross_reference → assess → validate
```
Each node's output feeds the next — equivalent to CoT reasoning steps but with explicit state, not implicit in a single prompt. This makes reasoning auditable and each step independently testable.

### Few-Shot Templates
Current implementation uses zero-shot prompts. The retrieval context acts as implicit few-shot examples — the LLM sees real regulatory text before answering. Production upgrade: add 1–2 worked compliance examples to `assess_risk` prompt for complex multi-framework scenarios.

---

## 14. Document Version Control in Vector Store

When a regulatory document is updated (e.g. a new RBI circular supersedes an older one):

**Upsert strategy:**
```sql
-- chunks table has UNIQUE(doc_id, section_id)
-- Re-ingesting with same doc_id + section_id overwrites the chunk
INSERT INTO chunks (...) VALUES (...)
ON CONFLICT (doc_id, section_id) DO UPDATE SET
    text = EXCLUDED.text,
    embedding = EXCLUDED.embedding,
    tsv = EXCLUDED.tsv,
    version = EXCLUDED.version;
```

**Version tracking:**
- `version` field on every chunk (e.g. `"2024"`, `"2025"`)
- Old version chunks remain if `doc_id` changes (e.g. `KYC2024` → `KYC2025`)
- `analyze_change_impact(new_doc_id)` surfaces which old sections overlap with the new document

**Amendment workflow:**
```
1. Download new circular → place in data/
2. python -m src.ingestion.run  (upserts new chunks, preserves old doc if different doc_id)
3. POST /analyze/change-impact {"doc_id": "new_doc_id"}  → see what changed
4. Review affected sections + transaction types
```

**Known limitation:** No automated diff between old and new versions of the same document. Production would add a `supersedes` metadata field and a `is_active` flag.

---

## 15. Guardrails — Full Implementation Map

### Hallucination Prevention (Pre-generation)
| Guardrail | Where | Mechanism |
|---|---|---|
| Retrieval grounding | `src/rag/answer.py` | Prompt: "use ONLY the regulatory text below" |
| Confidence gate | `src/rag/answer.py:83` | cosine < 0.3 → return "Insufficient context" |
| Sufficiency gate | `src/agent/nodes.py:134` | ≥2 fields missing → `flag_for_review` |
| Confidence gate (agent) | `src/agent/nodes.py:213` | Low retrieval → `flag_for_review` |

### Output Validation (Post-generation)
| Guardrail | Where | Mechanism |
|---|---|---|
| Pydantic API validation | `src/api/main.py` | All request/response schemas validated |
| Agent schema validation | `validate_output` node | `ComplianceAssessment` Pydantic parse |
| Citation requirement | `validate_output` node | Assessed result must carry ≥1 citation |
| JSON best-effort extraction | `_parse_llm_json()` | Regex fallback if LLM wraps JSON in text |

### PII Redaction (Input sanitisation)
| Guardrail | Where | Mechanism |
|---|---|---|
| PII detection + redaction | `src/security/pii_redactor.py` | Regex patterns: Aadhaar, PAN, phone, email, account, IFSC |
| Applied before storage | `src/api/main.py` | `redact(scenario)` called before `log_transaction()` |
| Warning logged | `src/api/main.py` | PII types found are logged at WARNING level |

### Regulatory Accuracy
| Guardrail | Where | Mechanism |
|---|---|---|
| Source citations required | All answers | Every answer carries `{doc_id, section_id, version}` |
| Retrieval threshold | `config.py: 0.3` | Low-quality chunks never reach LLM |
| Faithfulness eval | `src/eval/scorer.py` | FaithfulnessMetric scores answer vs context |
| **Post-hoc runtime check** | ❌ Not implemented | Production: run faithfulness check per query; flag score < 0.5 |

---

## 16. Production Cloud Architecture (AWS)

### 16.1 Kubernetes Topology (AWS EKS)

```
┌──────────────────────────── AWS EKS Cluster ─────────────────────────────┐
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Namespace: finserv-compliance                                       │  │
│  │                                                                      │  │
│  │  ┌──────────────────┐    ┌──────────────────┐                       │  │
│  │  │  fastapi-api      │    │  fastapi-api      │  ← HPA: 2–10 pods   │  │
│  │  │  (CPU: 0.5 core)  │    │  (CPU: 0.5 core)  │    CPU trigger >60% │  │
│  │  └────────┬─────────┘    └────────┬──────────┘                      │  │
│  │           └──────────┬────────────┘                                  │  │
│  │                      │ internal service :8000                        │  │
│  │                      ▼                                               │  │
│  │  ┌───────────────────────────────────┐                               │  │
│  │  │  ALB Ingress (HTTPS, WAF enabled) │ ← SSL termination             │  │
│  │  └───────────────────────────────────┘                               │  │
│  │                                                                      │  │
│  │  ┌──────────────────────────────────┐                                │  │
│  │  │  ollama-inference                │  ← 1 pod per GPU node          │  │
│  │  │  Node: g4dn.xlarge (T4 GPU)      │    model: llama3:70b           │  │
│  │  │  GPU: 1x NVIDIA T4 (16GB)        │    loaded in-memory            │  │
│  │  └──────────────────────────────────┘                                │  │
│  │                                                                      │  │
│  │  ┌──────────────────────────────────┐                                │  │
│  │  │  embedding-service               │  ← BGE-small CPU pod           │  │
│  │  │  (CPU: 1 core, Memory: 2GB)      │    HPA: 1–4 pods               │  │
│  │  └──────────────────────────────────┘                                │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌───────────────────┐   ┌───────────────────┐   ┌──────────────────┐    │
│  │  RDS PostgreSQL   │   │  ElastiCache Redis │   │  S3 (PDF store)  │    │
│  │  + pgvector ext.  │   │  (session cache)   │   │  (raw documents) │    │
│  │  Multi-AZ standby │   └───────────────────┘   └──────────────────┘    │
│  └───────────────────┘                                                    │
└────────────────────────────────────────────────────────────────────────────┘
```

### 16.2 Auto-Scaling Strategy

| Component | Scaling Trigger | Min | Max | Notes |
|---|---|---|---|---|
| FastAPI pods | CPU > 60% | 2 | 10 | Stateless — scale freely |
| Embedding service | CPU > 70% | 1 | 4 | BGE-small is CPU-bound |
| Ollama inference | Queue depth > 5 | 1 | 3 | GPU nodes expensive — scale conservatively |
| RDS | Read replicas | 1 | 3 | Add read replicas for retrieval load |

**Ollama GPU scaling rationale:** Each GPU node costs ~$0.526/hr (g4dn.xlarge). A single T4 handles ~10 concurrent requests at 2–3s latency with llama3:70b. At 10K queries/day peak (~7 QPS), 2 GPU nodes suffice with headroom.

### 16.3 Cost Estimation (500 users, 10K queries/day)

**Assumptions:**
- 10K queries/day = ~0.12 QPS average, ~7 QPS peak (8h business window)
- Mix: 70% RAG queries, 30% agent assessments
- Each query: 1 embedding call + 1 LLM call (avg 2s on GPU)

| Component | Spec | Monthly Cost (USD) |
|---|---|---|
| EKS cluster (control plane) | — | $73 |
| FastAPI pods (2× t3.medium) | 2 vCPU, 4GB each | $60 |
| Ollama GPU nodes (2× g4dn.xlarge) | 4 vCPU, 16GB, T4 GPU | $760 |
| Embedding service (2× t3.small) | 2 vCPU, 2GB each | $30 |
| RDS PostgreSQL (db.t3.medium, Multi-AZ) | 2 vCPU, 4GB, 100GB SSD | $120 |
| ALB + data transfer | — | $25 |
| S3 (PDF storage, ~10GB) | — | $2 |
| CloudWatch monitoring | — | $20 |
| **Total** | | **~$1,090/month** |

**Cost optimisation levers:**
- Use Spot instances for embedding service: saves ~40% → ~$940/month
- Use Graviton (ARM) for FastAPI pods: saves ~20% on compute
- Cache frequent queries in Redis: reduces LLM calls by ~30%
- Schedule GPU nodes only during business hours (8AM–8PM): saves ~50% on GPU cost → ~$760/month

---

## 17. Observability

### 17.1 Monitoring Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Observability Stack                        │
│                                                              │
│  FastAPI ──► Prometheus metrics ──► Grafana dashboards       │
│                   │                                          │
│              LangFuse (self-hosted) ──► LLM tracing          │
│                   │                                          │
│              CloudWatch Logs ──► Alerting                    │
└─────────────────────────────────────────────────────────────┘
```

**Metrics collected:**

| Metric | Tool | Alert threshold |
|---|---|---|
| API request latency (p50/p95/p99) | Prometheus + Grafana | p95 > 10s |
| LLM call latency per node | LangFuse | > 30s |
| Retrieval score distribution | Custom metric | mean cosine < 0.4 |
| `needs_review` rate | Prometheus counter | > 20% of requests |
| PII detection rate | Prometheus counter | Spike > 5% |
| Faithfulness score (eval runs) | eval_report.md | Drop > 10% from baseline |
| Token usage per query | LangFuse | Budget alert |

**LangFuse integration** (self-hosted, open-source):
```python
# Wraps Ollama calls to capture traces
from langfuse import Langfuse
langfuse = Langfuse(host="http://langfuse-service:3000")
# Traces: input, output, latency, model, tokens per node
```

### 17.2 Drift Detection

Retrieval relevance degrades when:
- New regulatory amendments change terminology
- Query patterns shift over time

**Strategy:**

```
Weekly scheduled job:
  1. Run run_eval() on golden dataset (data/eval_set.json)
  2. Compute 4 RAGAS metrics
  3. Compare against baseline (stored in eval_report.md)
  4. If faithfulness drops >10% OR context_recall drops >15%:
     → Alert compliance team
     → Trigger re-ingestion review
     → Consider re-embedding with updated model
```

**Baseline tracking:**
```python
# eval_report.md stores dated baselines:
# 2026-06-11: faithfulness=0.78, recall=0.71
# Alert if next run: faithfulness < 0.70 or recall < 0.60
```

---

## 18. Security — Extended

### 18.1 Encryption

| Layer | Mechanism | Status |
|---|---|---|
| Data at rest (Postgres) | AES-256 via RDS encryption (AWS KMS) | Production |
| Data in transit (API) | TLS 1.3 via ALB (AWS Certificate Manager) | Production |
| Data in transit (internal) | mTLS between pods (Istio service mesh) | Production |
| Model weights at rest | Encrypted EBS volumes | Production |
| Prototype (local) | No encryption (localhost only) | Prototype — acceptable |

### 18.2 Access Controls

| Control | Mechanism |
|---|---|
| API authentication | JWT tokens via AWS Cognito (production) |
| Database access | IAM database authentication (RDS) |
| Model endpoint | Internal cluster network only (no external exposure) |
| Audit log access | Read-only IAM role for compliance officers |
| PDF source documents | S3 bucket policy — write once, read many |

### 18.3 Data Sovereignty (enforced in code)
```python
# src/eval/scorer.py — runtime assertion
assert not any(ext in settings.ollama_base_url
               for ext in ["api.openai.com", "api.anthropic.com", "openai.azure.com"])
```
No regulated text ever leaves the cluster. All LLM inference is intra-cluster.

---

## 19. Fallback and Cost Controls

| Control | Location | Effect |
|---|---|---|
| Retrieval score threshold (0.3) | `config.py` | Low-quality queries never reach LLM |
| Sufficiency gate (≥2 missing fields) | `nodes.py` | Ambiguous input → `needs_review` |
| Confidence gate (cosine < threshold) | `nodes.py` | Low retrieval confidence → `needs_review` |
| Pydantic output validation | `validate_output` node | Malformed LLM output → `needs_review` |
| Regex amount fallback | `parse_input` node | LLM parse failure → regex extraction |
| LLM timeout (120s) | `config.py` | Hung Ollama call killed |
| Rerank top-n (4 chunks) | `config.py` | Limits context window cost |
| DeepEval retry cap (2 attempts) | `DEEPEVAL_RETRY_MAX_ATTEMPTS=1` | Judge failures don't loop indefinitely |
| Local inference only | All modules | Zero API cost, no rate limits |
