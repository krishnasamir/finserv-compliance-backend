# ADR-001: PostgreSQL + PGVector as the Vector Store

**Status:** Accepted  
**Date:** 2026-06-11  
**Deciders:** Architecture team  

---

## Context

The compliance assistant needs to store ~600 regulatory text chunks with:
- Dense vector embeddings for semantic search
- Full-text keyword index for hybrid retrieval
- Metadata filtering (by `framework`, `version`, `doc_id`)
- ACID guarantees (no partial ingestion)

We evaluated three options:

| Option | Pros | Cons |
|---|---|---|
| **PostgreSQL + PGVector** | Single store, ACID, metadata filtering, hybrid search, zero extra infra | Slower than specialist DBs at >100M vectors |
| Pinecone / Weaviate | Fast, managed, purpose-built | Proprietary, data leaves the machine (violates sovereignty constraint), extra service to operate |
| Chroma (local) | Open-source, simple | No SQL, no keyword index, no ACID, separate process |

---

## Decision

Use **PostgreSQL + PGVector** (Docker) as the sole data store for both vector embeddings and keyword (tsvector) index.

---

## Alternatives Considered and Rationale

1. **Data sovereignty** — regulated text never leaves the local machine. PGVector runs in Docker, same host as Ollama.
2. **Single ACID store** — ingestion is transactional. A failed PDF load cannot leave partial chunks in the index.
3. **Hybrid search** — `tsvector` keyword index + PGVector cosine index in the same query, fused via Reciprocal Rank Fusion. No second service needed.
4. **Metadata filtering** — standard SQL `WHERE framework = 'RBI-KYC'` without any custom DSL.
5. **Production parity** — the same schema scales to production by pointing at a managed Postgres (RDS, Cloud SQL). The only change is the `DATABASE_URL` config value.
6. **Open-source** — fully open-source, no licence cost.

---

## Consequences

- **Positive:** Single Docker container, one connection string, standard SQLAlchemy ORM.
- **Positive:** Full SQL expressiveness for future analytics (e.g., count chunks per framework).
- **Negative:** At >10M vectors, a dedicated ANN index (HNSW via pgvector is supported) would need tuning. Not a concern at prototype scale (~600 chunks).
- **Negative:** Requires Docker; pure in-memory option (Chroma) would be easier for unit tests — mitigated by fixture-scoped test corpus.
