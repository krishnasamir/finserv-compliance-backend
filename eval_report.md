# Evaluation Report

**Date:** 2026-06-10 15:50 (partial — judge process terminated at Q8)
**Questions evaluated:** 11
**Judge model:** mistral:7b @ http://localhost:11434
**Total judge HTTP calls made:** 85
**Note:** Scores marked `0.0*` = judge outputted invalid JSON (failed after 2 attempts).
Scores marked `1.0†` = out-of-corpus shortcut (empty context + refusal = maximally faithful).
Scores marked `—` = judge call completed HTTP 200 but process killed before writing scores.
Q9 was not reached before termination.

---

## Summary Metrics (partial — based on confirmed data only)

| Metric             | Confirmed 0.0 | Shortcut 1.0 | Completed (value unk.) | Not reached |
|--------------------|:---:|:---:|:---:|:---:|
| Faithfulness       | Q3, Q7 | Q10, Q11 | Q1, Q2, Q4, Q5, Q6, Q8 | Q9 |
| Answer Relevance   | Q1, Q2, Q3, Q5, Q6, Q7 | — | Q4, Q8, Q10, Q11 | Q9 |
| Context Precision  | Q2, Q6, Q8 | Q10, Q11 | Q1, Q3, Q4, Q5, Q7 | Q9 |
| Context Recall     | — | Q10, Q11 | Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8 | Q9 |

> **Judge reliability issue:** `mistral:7b` on CPU frequently outputs JSON missing the `verdict` field,
> causing DeepEval to raise a validation error after 2 attempts. ~13% failure rate across 85 calls.
> Production use should point to a 70B model via Ollama for reliable JSON output.

---

## Per-Question Detail

| # | Question | Framework | Answer (truncated) | Ctx | Citations | Faith | Rel | Prec | Rec |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | What is Know Your Customer (KYC) and what purpose does it serve? | RBI-KYC | Know Your Customer (KYC) is a process by which a Regulated Entity, including a bank… | 4 | 4 | — | 0.0* | — | — |
| 2 | When is a Regulated Entity required to carry out KYC under RBI guidelines? | RBI-KYC | A Regulated Entity is required to carry out KYC at the following times under RBI… | 4 | 4 | — | 0.0* | 0.0* | — |
| 3 | What documents are required for opening a bank account by an individual under RBI KYC rules? | RBI-KYC | The documents required for opening a bank account by an individual under RBI KYC… | 4 | 4 | 0.0* | 0.0* | — | — |
| 4 | What is a 'Small Account' under RBI KYC rules and how long can it operate? | RBI-KYC | A 'Small Account' under RBI KYC rules is a savings account opened in terms of… | 4 | 1 | — | — | — | — |
| 5 | How often must periodic updation of KYC (re-KYC) be carried out for different customer risk categories? | RBI-KYC | The periodic updation of KYC must be carried out as follows for different risk categories… | 4 | 4 | — | 0.0* | — | — |
| 6 | What is the Video-based Customer Identification Process (V-CIP) under RBI KYC rules? | RBI-KYC | The Video-based Customer Identification Process (V-CIP) is an alternate method of… | 4 | 4 | — | 0.0* | 0.0* | — |
| 7 | Are retained earnings part of Common Equity Tier 1 capital under Basel III? | Basel III | Retained earnings are part of Common Equity Tier 1 capital under Basel III. They… | 4 | 1 | 0.0* | 0.0* | — | — |
| 8 | What features are considered an incentive to redeem for Additional Tier 1 capital instruments under Basel III? | Basel III | The following are considered an incentive to redeem for Additional Tier 1 capital… | 4 | 4 | — | — | 0.0* | — |
| 9 | What is the minimum trigger level for write-down or conversion of Additional Tier 1 instruments under Basel III? | Basel III | The minimum trigger level for write-down or conversion of Additional Tier 1 (AT1)… | 4 | 1 | — | — | — | — |
| 10 | Under MiFID II, what is the appropriateness assessment required before selling a complex product? | none (OOC) | Insufficient context: The given regulatory text does not provide specific information… | 0 | 0 | 1.0† | — | 1.0† | 1.0† |
| 11 | What provisions of the Glass-Steagall Act separated commercial and investment banking activities? | US-Law (OOC) | Insufficient context: The Glass-Steagall Act and its provisions regarding separation… | 0 | 0 | 1.0† | — | 1.0† | 1.0† |

---

## Failure Analysis

### Judge model failures (invalid JSON output)
`mistral:7b` on CPU frequently omits the `verdict` field in structured JSON responses. 11 confirmed
failures across 85 total judge HTTP calls (~13% failure rate). These are scored as `0.0`.

**Affected questions:**
- Q1: answer_relevance
- Q2: context_precision, answer_relevance
- Q3: faithfulness, answer_relevance
- Q5: answer_relevance
- Q6: context_precision, answer_relevance
- Q7: faithfulness, answer_relevance
- Q8: context_precision

### Out-of-corpus refusals (correct behaviour)
- **Q10 — MiFID II**: System correctly declined. No MiFID II document ingested.
- **Q11 — Glass-Steagall**: System correctly declined. US banking law not in corpus.
