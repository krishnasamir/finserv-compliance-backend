# Claude Code — Phase-by-Phase Prompts

Paste these one at a time into Claude Code, in order. Wait for each phase to be green before moving on. Claude Code auto-loads `CLAUDE.md`; you point it at one phase spec per phase.

---

## Phase 0 — One-time setup (do this yourself, before Claude Code)

```bash
# 1. Install Ollama, then pull the model
ollama pull qwen2.5:3b

# 2. Start Postgres + PGVector
cp .env.example .env
docker compose up -d
docker compose ps          # confirm healthy

# 3. Python env
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 4. Put 5+ regulatory PDFs in ./data/
#    (RBI master circulars from rbi.org.in, Basel III excerpts from bis.org)
```

Then open Claude Code in the repo folder.

---

## Prompt 1 — Project scaffold

```
Read CLAUDE.md. Scaffold the project structure: src/{ingestion,retrieval,rag,agent,eval,api}/, tests/, config.py loading from .env, requirements.txt with the pinned stack, and a pytest setup. Do not implement features yet — just the skeleton, config loader, and a conftest.py. Then run pytest to confirm the empty suite collects.
```

Verify: `pytest` runs, config loads from `.env`.

---

## Prompt 2 — Phase 2A, tests first

```
Read docs/2A-rag-pipeline.md. Following the TDD rule in CLAUDE.md, write the acceptance tests (1–6) for the RAG pipeline FIRST, in tests/. Use a small fixture corpus so tests don't depend on my real PDFs. Run pytest and confirm they fail for the right reasons. Do not write implementation yet.
```

Verify: tests exist and fail (not error) — they should fail because modules are unimplemented.

---

## Prompt 3 — Phase 2A, implement to green

```
Now implement src/ingestion, src/retrieval, and src/rag to make the 2A acceptance tests pass, one module at a time (ingestion → retrieval → rag). Run pytest after each module. Keep everything config-driven and modular per CLAUDE.md. Handle the error paths in test 6 explicitly. Stop when all 2A tests are green.
```

Then ingest your real documents:
```
Add a CLI entrypoint `python -m src.ingestion.run` that ingests all PDFs in DATA_DIR. Run it on my ./data folder and show me the chunk count and a sample of the stored metadata.
```

Verify: ask 2–3 real questions and confirm cited answers.

---

## Prompt 4 — Phase 2B, tests first

```
Read docs/2B-agent.md. Phase 2A is green. Following TDD, write the 2B acceptance tests (1–6) FIRST, including the $2M example and the ambiguous-input case. Run pytest and confirm they fail. Do not implement the agent yet.
```

Verify: tests fail cleanly.

---

## Prompt 5 — Phase 2B, implement to green

```
Implement the LangGraph agent in src/agent to pass the 2B tests: typed AgentState, the nodes and two gates from the spec, the Pydantic ComplianceAssessment output, and the flag_for_review path. Reuse the 2A pipeline for retrieval — no duplicate retrieval logic. Run pytest until all 2B tests are green.
```

Verify: run the $2M example end to end; confirm structured output with citations, and that "a payment was made" returns needs_review.

---

## Prompt 6 — Phase 2C, dataset then tests then implement

```
Read docs/2C-evaluation.md. First scaffold data/eval_set.json with 3 example pairs so I can see the format — I will fill in the full 15–20 myself. Then, following TDD, write the 2C acceptance tests FIRST and confirm they fail.
```

Fill in your 15–20 eval pairs in `data/eval_set.json` yourself, then:
```
Implement src/eval to pass the 2C tests: run_eval, score (RAGAS with the judge model pointed at the local Ollama endpoint per config — assert it's not an external URL), and report writing eval_report.md. Run pytest until green, then run the full evaluation on my eval_set.json and show me eval_report.md.
```

Verify: `eval_report.md` exists with four metric averages + failure analysis.

---

## Prompt 7 — API + README polish

```
Add a FastAPI app in src/api exposing POST /query (RAG) and POST /assess (agent). Then write README.md: project overview, the architecture diagrams from my docs, setup instructions (Phase 0 steps), how to run ingestion/query/assess/eval, sample inputs and outputs, and a "Design decisions" section linking choices to the ADRs (PGVector, local model, LangGraph). Add .gitignore (exclude .env, .venv, data/*.pdf, __pycache__).
```

Verify: `uvicorn` serves both endpoints; README is complete.

---

## Final checklist before submitting

- [ ] `pytest` — all phases green
- [ ] `docker compose up -d` + ingestion runs clean from scratch
- [ ] `/query` returns cited answers; `/assess` returns structured assessment
- [ ] Ambiguous input → needs_review (not fabricated)
- [ ] `eval_report.md` present with scores + failure analysis
- [ ] README with diagrams, setup, samples, design rationale
- [ ] `.env` excluded from git; `.env.example` included
- [ ] No hardcoded model names; no external LLM API calls anywhere

---

## Tips while using Claude Code

- If it tries to write code before tests, stop it: "Tests first per CLAUDE.md."
- If it reaches for a proprietary service, stop it: "Open-source only — use the local substitute."
- Review each module before moving on — you must be able to defend every choice in the interview.
- Keep phases isolated: don't start 2B until 2A is fully green.
- Commit after each green phase so you have clean checkpoints.
