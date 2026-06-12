# ADR-002: Local Ollama for All LLM Inference

**Status:** Accepted  
**Date:** 2026-06-11  
**Deciders:** Architecture team  

---

## Context

The compliance assistant uses an LLM for three tasks:
1. Answer generation (RAG pipeline)
2. Compliance risk assessment (agent nodes)
3. Evaluation judge (DeepEval scoring)

We evaluated two inference approaches:

| Option | Pros | Cons |
|---|---|---|
| **Local Ollama** | Data sovereignty, zero cost, no rate limits, offline capable | Slower on CPU, smaller models available locally |
| OpenAI / Anthropic API | Fast, large models, high JSON reliability | Regulated text sent to external server (sovereignty violation), per-token cost, rate limits, internet dependency |

---

## Decision

All LLM inference — answer generation, agent reasoning, and evaluation judge — runs locally via **Ollama** on `http://localhost:11434`. No regulated text is ever sent to an external API.

The model name is a **single config value** (`llm_model` in `config.py`). Swapping from the 3B prototype model to a production 70B model requires changing one line.

---

## Alternatives Considered and Rationale

1. **Data sovereignty (hard constraint)** — Basel III and RBI KYC text are regulated financial documents. Sending them to a hosted API introduces legal and compliance risk. Local inference eliminates this entirely.
2. **Zero inference cost** — no per-token billing. Running 600 RAG queries costs the same as running 1.
3. **No rate limits** — the evaluation suite makes ~85 sequential LLM calls. A hosted API would throttle or bill significantly.
4. **Auditable** — all inference is local, logged, and reproducible. No external dependency that can change behaviour between runs.
5. **Production upgrade path** — the prototype uses `qwen2.5:3b` (1.9 GB). Production swaps to a 70B model via the same Ollama interface. The code is unchanged.
6. **Open-source** — Ollama and all supported models are open-source.

---

## Model Selection

| Model | Use | Size | Notes |
|---|---|---|---|
| `qwen2.5:3b` | RAG + agent inference | 1.9 GB | Fast on CPU; sufficient for retrieval-grounded answers |
| `mistral:7b` | Evaluation judge | 4.4 GB | Better structured JSON output than 3B for DeepEval metrics |

---

## Consequences

- **Positive:** Full data sovereignty. No API keys, no network dependency, no cost.
- **Positive:** Model name is config-driven — `llm_model: str = "qwen2.5:3b"` in `config.py`. Production swap = one line.
- **Negative:** CPU inference is slow (~15–30s per call). Mitigated by: session-scoped test fixtures (avoid redundant calls), retrieval gate (low-quality queries never reach the LLM).
- **Negative:** Small local models (3B–7B) occasionally produce malformed JSON. Mitigated by: `_parse_llm_json()` best-effort extractor, Pydantic validation at every boundary, `needs_review` fallback.
- **Assertion in code:** `scorer.py` asserts `ollama_base_url` does not contain `api.openai.com`, `api.anthropic.com`, or `openai.azure.com` — enforces the sovereignty constraint at runtime.
