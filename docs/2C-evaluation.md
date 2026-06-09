# Phase 2C — Evaluation Framework (10 points)

Automated quality scoring of the RAG pipeline. Depends on 2A (and ideally 2B) being complete. Build last.

## Goal

Score the pipeline on a golden dataset using faithfulness, answer relevance, context precision, and context recall; produce a summary report with scores and failure analysis.

## Components

### `data/eval_set.json`
15–20 hand-written `{question, ground_truth, expected_sources}` pairs derived from the ingested corpus. Written manually (ground-truth quality matters). Cover: single-framework lookups, cross-framework questions, and at least 2 questions whose answer is "not in the corpus" (to test honest refusal).

### `src/eval/`
- `run_eval(eval_set) -> Results` — for each question, run the 2A `answer()`, collect (question, answer, contexts, ground_truth).
- `score(results) -> Metrics` — compute faithfulness, answer relevance, context precision, context recall using RAGAS (or DeepEval). **Judge model = local Ollama** (open-source constraint). Embeddings = local BGE.
- `report(metrics, results) -> None` — write `eval_report.md`: per-metric average scores, a per-question score table, and a failure analysis section listing the lowest-scoring questions with a one-line hypothesis for each (e.g. "low context recall — chunk boundary split the clause").

## Acceptance tests (write these first)

1. `eval_set.json` loads and contains 15–20 well-formed pairs.
2. `run_eval` produces one result row per question with non-null answer and contexts.
3. `score` returns all four metrics as floats in [0,1]; the judge model call points at the local Ollama endpoint (assert config, not an external URL).
4. `report` writes `eval_report.md` containing the four metric averages and a failure-analysis section.
5. The "not in corpus" questions score high on faithfulness only if the system correctly refused (sanity check that refusal is rewarded, not penalized as a wrong answer).

## Definition of done

All acceptance tests green; `eval_report.md` generated with scores + failure analysis; judge model is local (no external API); the report is reproducible by a single command.
