# Phase 2A — RAG Pipeline (20 points)

Functional RAG pipeline over regulatory documents. Build modules in the order below; write acceptance tests first per the TDD rule in `CLAUDE.md`.

## Goal

Ingest 5+ regulatory PDFs, store them in PGVector, and answer natural-language questions with hybrid search + re-ranking, returning cited, source-attributed answers.

## Modules & contracts

### `src/ingestion/`
- `parse_pdf(path) -> list[Page]` — extract text per page, preserving section/clause structure. PyMuPDF; fall back gracefully on malformed pages.
- `chunk(pages, framework, version) -> list[Chunk]` — clause/section-aware chunking, target ~512 tokens, 10–15% overlap. Each `Chunk` carries metadata: `doc_id, framework, version, effective_date, section_id, text`.
- `embed(chunks) -> list[Chunk]` — add BGE-small embedding vector to each chunk.
- `load(chunks)` — upsert into PGVector; populate both the vector column and a `tsvector` keyword column.

### `src/retrieval/`
- `dense_search(query, k) -> list[ScoredChunk]` — PGVector cosine top-k.
- `keyword_search(query, k) -> list[ScoredChunk]` — Postgres full-text (tsvector) top-k.
- `hybrid_search(query, k) -> list[ScoredChunk]` — run both, fuse with reciprocal rank fusion, dedupe.
- `rerank(query, chunks) -> list[ScoredChunk]` — BGE-reranker-base cross-encoder, reorder by relevance.
- `compress(query, chunks) -> list[Chunk]` — trim each chunk to query-relevant spans.

### `src/rag/`
- `answer(query) -> Answer` — orchestrates retrieve → rerank → compress → generate via Ollama. Returns `Answer{text, citations: list[Citation]}` where each `Citation` resolves to `{doc_id, section_id, version}`.

## Acceptance tests (write these first)

1. Ingesting the sample corpus produces > 0 chunks, each with non-null framework/section metadata.
2. `dense_search` and `keyword_search` each return results for a known in-corpus term.
3. `hybrid_search` returns results containing items that keyword-only misses (semantic) and items dense-only misses (exact term, e.g. a section number).
4. `rerank` changes order vs. pre-rerank for at least one query (precision improvement).
5. `answer("known question")` returns non-empty text **and** at least one citation that resolves to a real ingested chunk.
6. Error paths: empty query raises a typed error; a query with no matches returns an "insufficient context" answer (not a crash, not a hallucinated answer).

## Definition of done

All acceptance tests green; modules independently testable; no hardcoded model names or paths (all from config); empty-retrieval and timeout paths handled.
