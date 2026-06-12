# FinServ Compliance Assistant

Regulatory compliance RAG + agentic checker over Basel III, MiFID II, and RBI regulatory text.
Answers natural-language compliance questions with cited sources and assesses transaction scenarios
against ingested regulations — all inference runs **locally via Ollama** (data sovereignty, zero cost).

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              FastAPI (src/api)           │
                        │   POST /query          POST /assess      │
                        └────────┬───────────────────┬────────────┘
                                 │                   │
                    ┌────────────▼──────┐   ┌────────▼──────────────┐
                    │  RAG Pipeline     │   │  LangGraph Agent      │
                    │  (src/rag)        │   │  (src/agent)          │
                    │                   │   │                       │
                    │  retrieve         │   │  parse_input          │
                    │  rerank           │   │  check_sufficiency ──►flag_for_review
                    │  compress         │   │  classify_frameworks  │
                    │  generate         │   │  retrieve             │
                    └────────┬──────────┘   │  check_confidence ──►flag_for_review
                             │              │  cross_reference      │
                             │              │  assess_risk          │
                             │              │  validate_output      │
                             │              └────────┬──────────────┘
                             │                       │
                    ┌────────▼───────────────────────▼────────────┐
                    │           src/retrieval                      │
                    │   dense_search + keyword_search → hybrid RRF │
                    │           + BGE reranker                     │
                    └────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │     PostgreSQL + PGVector (Docker)           │
                    │   vector index (cosine)  tsvector keyword    │
                    └────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │           src/ingestion                      │
                    │   parse_pdf → chunk → embed → load           │
                    │   PyMuPDF | BAAI/bge-small-en-v1.5          │
                    └─────────────────────────────────────────────┘

                    LLM inference: Ollama (local) — qwen2.5:3b / swap to 70B for prod
                    Eval judge:    Ollama (local) — mistral:7b
```

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | Project requirement |
| LLM | Ollama `qwen2.5:3b` (local) | Data sovereignty; model name is config-only |
| Embeddings | `BAAI/bge-small-en-v1.5` | Local, high quality for retrieval |
| Reranker | `BAAI/bge-reranker-base` | Local cross-encoder; improves precision |
| Vector store | PostgreSQL + PGVector | Single ACID store; metadata filtering; hybrid search |
| Agent framework | LangGraph | Deterministic, auditable control flow |
| Validation | Pydantic | Typed I/O, catches LLM hallucination before it escapes |
| Eval | DeepEval (local judge) | RAGAS-style metrics; judge = local Ollama |
| API | FastAPI | Auto OpenAPI docs; type-safe request/response |
| Tests | pytest | TDD throughout |

---

## Setup (Phase 0)

### 1. Install Ollama and pull the model

```bash
# Install Ollama from https://ollama.com
ollama pull qwen2.5:3b        # RAG answer generation
ollama pull mistral:7b        # eval judge
```

### 2. Start PostgreSQL + PGVector

```bash
cp .env.example .env          # edit DATABASE_URL if needed
docker compose up -d
docker compose ps             # confirm: finserv_pgvector  Up (healthy)
```

### 3. Create Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Place regulatory PDFs in `./data/`

Obtain these manually (no scraping by code):
- RBI KYC Master Direction — [rbi.org.in](https://www.rbi.org.in)
- Basel III rules text (bcbs211, d424, defcap_b3) — [bis.org](https://www.bis.org)

Expected files:
```
data/KYC09062025.pdf
data/MD18KYCF6E92C82E1E1419D87323E3869BC9F13.pdf
data/bcbs211.pdf
data/bcbs221.pdf
data/d424.pdf
data/defcap_b3.pdf
```

---

## Running the system

### Ingest documents

```bash
python -m src.ingestion.run
# Output: chunk count, sample metadata
# ~599 chunks from 6 PDFs
```

### Start the API server

```bash
uvicorn src.api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### Query the RAG pipeline

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is KYC and what purpose does it serve?"}' | python3 -m json.tool
```

### Run a compliance assessment

```bash
curl -s -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{"scenario": "Cross-border payment of $2M to a non-KYC entity in a high-risk jurisdiction"}' \
  | python3 -m json.tool
```

### Run the evaluation suite

```bash
# Requires data/eval_set.json (15–20 QA pairs)
python -m src.eval.runner          # or via pytest
pytest tests/eval/test_eval.py -v
# Report written to eval_report.md
```

### Run all tests

```bash
pytest
```

---

## Sample inputs and outputs

### POST /query

**Request:**
```json
{"question": "Are retained earnings part of Common Equity Tier 1 capital under Basel III?"}
```

**Response:**
```json
{
  "answer": "Retained earnings are part of Common Equity Tier 1 capital under Basel III. They are included in the positive components of CET1 along with other disclosed reserves as stated on the balance sheet. To arrive at the final CET1 figure, these positive components are then reduced by the regulatory adjustments set out in paragraphs 66 to 90 of the Basel III rules text. [REF-1][REF-2]",
  "citations": [
    {"doc_id": "bcbs211", "section_id": "para_52", "version": "2011-12"},
    {"doc_id": "defcap_b3", "section_id": "cet1_components", "version": "2010-12"}
  ]
}
```

### POST /assess

**Request:**
```json
{"scenario": "A bank onboards a new retail customer without collecting any identity documents."}
```

**Response:**
```json
{
  "status": "assessed",
  "risk_rating": "high",
  "applicable_regulations": ["RBI-KYC"],
  "required_actions": [
    "Collect Officially Valid Document (OVD) showing name and address",
    "Obtain PAN or Form 60",
    "Complete Customer Due Diligence before activating account"
  ],
  "citations": [
    {"doc_id": "KYC09062025", "section_id": "sec_16", "version": "2025-06"},
    {"doc_id": "MD18KYCF6E92C82E1E1419D87323E3869BC9F13", "section_id": "para_4", "version": "2024-01"}
  ]
}
```

**Ambiguous input → graceful degradation:**
```json
// Request:  {"scenario": "a payment was made"}
// Response:
{
  "status": "needs_review",
  "risk_rating": "low",
  "applicable_regulations": [],
  "required_actions": ["Assessment failed. Human review required."],
  "citations": []
}
```

---

## Evaluation

The evaluation framework (`src/eval/`) implements 4 RAGAS-style metrics via DeepEval with a local Ollama judge:

| Metric | Description |
|---|---|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevance | Does the answer address the question? |
| Context Precision | Are the retrieved chunks relevant to the question? |
| Context Recall | Does the retrieved context cover the ground truth? |

Results are written to `eval_report.md`. Out-of-corpus questions (MiFID II, Glass-Steagall) correctly trigger graceful degradation and receive faithfulness = 1.0.

---

## Design decisions

### PGVector as the vector store
A single PostgreSQL instance provides both the vector index (cosine similarity via PGVector extension) and a `tsvector` keyword index. This gives hybrid search (dense + BM25-style) with ACID guarantees, metadata filtering, and zero additional infrastructure — avoiding the operational overhead of a separate vector database (e.g. Pinecone, Weaviate). See `docs/2A-rag-pipeline.md`.

### Local Ollama for all inference
All LLM calls — answer generation, agent reasoning, and eval judge — route to Ollama running locally. This satisfies two hard constraints: (1) **data sovereignty** — regulated text (RBI/Basel III) never leaves the machine; (2) **open-source only** — no proprietary API. The model name is a single config value (`llm_model` in `config.py`); swapping from the 3B prototype to a production 70B model requires one line change. See `CLAUDE.md`.

### LangGraph for the compliance agent
LangGraph's explicit node-and-edge graph produces deterministic, auditable control flow. Two conditional gates (`check_sufficiency`, `check_confidence`) route ambiguous or low-confidence inputs to `flag_for_review` before any risk assessment is emitted. This makes the failure path first-class and traceable — not a fallback. See `docs/2B-agent.md`.

### Pydantic for all structured I/O
Every module boundary uses a Pydantic model (`Answer`, `ComplianceAssessment`, `AgentState`, API request/response schemas). This catches LLM hallucination (malformed JSON, missing fields) at the validation layer before it propagates. The `validate_output` agent node re-parses LLM output through `ComplianceAssessment`; failure routes to `flag_for_review`, never to a raw string response.

### Graceful degradation over fabrication
Empty retrieval, low confidence scores, and malformed LLM output all produce a `needs_review` response with `status="needs_review"` — never a fabricated answer. This is enforced at three layers: the retrieval gate (score threshold in `config.py`), the agent confidence gate, and the Pydantic output validator.
