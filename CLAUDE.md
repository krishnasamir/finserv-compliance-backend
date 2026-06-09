# CLAUDE.md — Project guide for Claude Code

Regulatory Compliance Assistant for FinServ Global. RAG + agentic compliance checker over Basel III, MiFID II, and RBI regulatory text. This file is loaded on every prompt — keep it lean. Detailed per-phase contracts live in `docs/` and are referenced when that phase begins.

## Working method: Test-Driven Development (mandatory)

Follow strict TDD on every feature:

1. Write the failing test(s) first, against the contract in the relevant `docs/*.md` spec.
2. Run the tests and confirm they fail for the right reason.
3. Write the minimum code to make them pass.
4. Refactor with tests green.
5. Never write implementation before its test exists. Never weaken a test to make code pass.

Run `pytest` after every change. A phase is "done" only when its spec's acceptance tests are green.

## Hard constraints (never violate)

- **Open-source only.** All models, libraries, and tools must be open-source. No proprietary or paid services in the runtime. If a proprietary option is the natural choice, use the open-source substitute and add a one-line code comment noting what it replaces.
- **Data sovereignty.** No regulated text is ever sent to an external/hosted LLM API. All inference is local via Ollama.
- **Traceability.** Every answer and assessment must carry source citations (document, section, version). No uncited claims.
- **Graceful degradation.** On insufficient input or low-confidence retrieval, return a clear "needs human review" result — never fabricate.

## Stack (fixed — do not substitute without asking)

- Language: Python 3.11+
- LLM: local via **Ollama**, model `qwen2.5:3b` (config-driven; production swaps to a 70B — model name is a config value, never hardcoded)
- Embeddings: `BAAI/bge-small-en-v1.5` via sentence-transformers (local)
- Reranker: `BAAI/bge-reranker-base` (local)
- Vector store: **PostgreSQL + PGVector** (Docker), with `tsvector` keyword index for hybrid search
- Agent framework: **LangGraph**
- Validation: **Pydantic** for all structured I/O
- Eval: **RAGAS** (or DeepEval), judge model pointed at local Ollama
- API: **FastAPI**
- Tests: **pytest**

## Repo conventions

- Modular: one responsibility per module. `src/ingestion/`, `src/retrieval/`, `src/agent/`, `src/eval/`, `src/api/`.
- Config-driven: all models, paths, thresholds in `config.py` / `.env` — no magic values in code.
- Every module has a matching `tests/` file.
- Errors are explicit: typed exceptions, no bare `except`. Empty retrieval, model timeout, and malformed input are handled paths, not crashes.
- Type hints everywhere. Docstrings on public functions.

## Build order (each phase depends on the previous)

1. **Phase 2A — RAG pipeline** → see `docs/2A-rag-pipeline.md`
2. **Phase 2B — Agentic compliance checker** → see `docs/2B-agent.md`
3. **Phase 2C — Evaluation framework** → see `docs/2C-evaluation.md`

Build 2A fully (tests green) before starting 2B. Build 2C last.

When starting a phase, read its spec file and write its acceptance tests first.

## Design rationale (for defense)

Key choices trace to the architecture decision records: PGVector (single ACID store, metadata filtering), local model (sovereignty + zero cost), LangGraph (deterministic, auditable control flow). The prototype is a faithful slice of the production design — same code, model name is the only thing that changes between local and production.

## Sample data

Place 5+ public regulatory PDFs in `data/` (RBI master circulars from rbi.org.in, Basel III excerpts from bis.org) before running ingestion. These are gathered manually, not fetched by code.
