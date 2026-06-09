"""Orchestrate retrieve → rerank → compress → generate to produce a cited answer."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from config import settings
from src.exceptions import EmptyQueryError

log = logging.getLogger(__name__)

_INSUFFICIENT_CONTEXT_TEXT = (
    "Insufficient context: no relevant regulatory provisions were found for this query. "
    "Please consult a qualified compliance expert or reformulate the question."
)

# Parse [REF-N] markers back from LLM output
_REF_RE = re.compile(r"\[REF-(\d+)\]")

_PROMPT_TEMPLATE = """\
You are a regulatory compliance expert. Answer the question using ONLY the \
regulatory text below.

REGULATORY CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Base your answer solely on the context above — never invent content.
2. Cite every factual claim using EXACTLY: [REF-N] where N matches the source \
number above. Example: "The minimum CET1 ratio is 4.5% [REF-1]."
3. If the context does not contain relevant information to answer this \
question, respond with ONLY: "Insufficient context: <one-sentence reason>"
   Do NOT include any [REF-N] citations in that case.

ANSWER:"""


@dataclass
class Citation:
    """Pointer to the exact source chunk that supports a claim."""

    doc_id: str
    section_id: str
    version: str


@dataclass
class Answer:
    """LLM-generated response with mandatory source citations."""

    text: str
    citations: list[Citation] = field(default_factory=list)


def answer(query: str) -> Answer:
    """Full RAG pipeline: retrieve → rerank → compress → generate via Ollama.

    Raises EmptyQueryError for blank or whitespace-only queries.
    Returns an 'insufficient context' Answer when retrieval yields nothing
    above the confidence threshold — never fabricates.
    """
    from src.retrieval.reranker import compress, rerank
    from src.retrieval.search import (
        dense_search, keyword_search, reciprocal_rank_fusion,
    )

    # ── Guard: empty / whitespace-only query ─────────────────────────────────
    if not query or not query.strip():
        raise EmptyQueryError(f"Query must be non-empty; got: {query!r}")

    # ── Single-pass: embed query once, run both search modes at full k ────────
    # Avoids the old pattern of a k=1 probe + hybrid_search, which re-embedded
    # the query and made 3 DB round-trips instead of 2.
    dense_results   = dense_search(query, k=settings.retrieval_top_k)
    keyword_results = keyword_search(query, k=settings.retrieval_top_k)

    # ── Relevance gate — raw cosine ONLY, NEVER RRF score ─────────────────────
    # RRF fusion scores ≈ 1/(rrf_k + rank) ≈ 0.016 for rank-1.  That is two
    # orders of magnitude below retrieval_score_threshold (default 0.3), so
    # comparing RRF scores against the threshold would refuse every valid query.
    # The gate MUST use the raw cosine score from dense_search.
    best_cosine = dense_results[0].score if dense_results else 0.0

    # A keyword hit also clears the gate, but ts_rank must be non-trivial to
    # avoid spurious partial-token matches inside compound strings such as
    # "xyzzy_nonexistent_regulation_clause_99999".
    has_quality_keyword = (
        len(keyword_results) > 0
        and keyword_results[0].score >= 0.05
    )

    if best_cosine < settings.retrieval_score_threshold and not has_quality_keyword:
        log.info(
            "Relevance gate CLOSED: cosine=%.3f < threshold=%.2f, kw_score=%s",
            best_cosine,
            settings.retrieval_score_threshold,
            keyword_results[0].score if keyword_results else "n/a",
        )
        return Answer(text=_INSUFFICIENT_CONTEXT_TEXT, citations=[])

    # ── RRF fusion — ranking only, NOT for threshold decisions ────────────────
    candidates = reciprocal_rank_fusion(
        [dense_results, keyword_results],
        k=settings.retrieval_top_k,
        rrf_k=settings.hybrid_rrf_k,
    )
    if not candidates:
        return Answer(text=_INSUFFICIENT_CONTEXT_TEXT, citations=[])

    reranked = rerank(query, candidates)[: settings.rerank_top_n]
    context_chunks = compress(query, reranked)

    # ── Build prompt with numbered REF labels ─────────────────────────────────
    ref_map: dict[int, object] = {}  # ref_number → Chunk
    blocks: list[str] = []
    for i, c in enumerate(context_chunks, start=1):
        ref_map[i] = c
        blocks.append(
            f"[REF-{i}] Source: {c.doc_id} | {c.framework} {c.version} | {c.section_id}\n{c.text}"
        )
    context_text = "\n\n".join(blocks)
    prompt = _PROMPT_TEMPLATE.format(context=context_text, query=query)

    # ── Generate via Ollama (all inference stays local) ───────────────────────
    try:
        import ollama as _ollama
        response = _ollama.chat(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": settings.llm_num_ctx, "temperature": 0.1},
        )
        answer_text: str = response["message"]["content"].strip()
    except Exception as exc:
        from src.exceptions import ModelTimeoutError
        log.error("Ollama call failed: %s", exc)
        raise ModelTimeoutError(f"LLM generation failed: {exc}") from exc

    # ── Extract citations — only refs the LLM actually used ───────────────────
    cited_nums = sorted(
        {int(m.group(1)) for m in _REF_RE.finditer(answer_text)},
    )
    citations: list[Citation] = []
    for n in cited_nums:
        if n in ref_map:
            c = ref_map[n]
            citations.append(Citation(doc_id=c.doc_id, section_id=c.section_id, version=c.version))

    # Fallback: small models (e.g. 3B) occasionally produce a correct answer
    # without emitting [REF-N] markers.  When the answer is substantive (not a
    # refusal), attribute every context chunk the LLM was given — they are the
    # actual source regardless of citation markup.
    is_refusal = "insufficient context" in answer_text.lower()[:60]
    if not citations and not is_refusal:
        citations = [
            Citation(doc_id=c.doc_id, section_id=c.section_id, version=c.version)
            for c in context_chunks
        ]

    return Answer(text=answer_text, citations=citations)
