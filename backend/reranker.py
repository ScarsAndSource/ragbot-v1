import logging
import cohere
from config import get_settings

settings = get_settings()
logger = logging.getLogger("ragbot.reranker")

# ── Lazy client initialization ────────────────────────────────────────────────
_cohere_client = None


def _get_client() -> cohere.Client | None:
    """
    Return a Cohere client, initializing on first call.
    Returns None if COHERE_API_KEY is not configured.
    """
    global _cohere_client
    if _cohere_client is None:
        if settings.cohere_api_key:
            _cohere_client = cohere.Client(api_key=settings.cohere_api_key)
            logger.info("Cohere reranker initialized — model: rerank-english-v3.0")
        else:
            logger.warning(
                "COHERE_API_KEY not set — reranker disabled. "
                "All queries will use raw hybrid-search order. "
                "Output guardrail will be disabled (no relevance scores available)."
            )
    return _cohere_client


def rerank_chunks(
    query: str,
    chunks: list[str],
    top_n: int | None = None,
) -> tuple[list[str], float]:
    """
    Rerank chunks using Cohere's cross-encoder model.

    Returns:
        (reranked_chunks, top_relevance_score)

        reranked_chunks    — top_n chunks sorted by Cohere relevance, best first
        top_relevance_score — Cohere's score for the best chunk (0.0 to 1.0)
                              Used by the output guardrail in chat.py to decide
                              whether the LLM had strong enough context to answer.

    Score interpretation:
        > 0.60  — strong context match, LLM should answer accurately
        0.20–0.60 — moderate match, answer likely grounded but may be incomplete
        < 0.20  — weak match, context is probably irrelevant to the query —
                  guardrail will override the LLM response

    Fallback score when Cohere unavailable or API fails:
        Returns 1.0 — fail open. Guardrail disabled rather than blocking all answers.
        Better to risk a thin answer than to block legitimate responses.
    """
    if top_n is None:
        top_n = settings.top_k_reranked

    # Edge case — no chunks
    if not chunks:
        logger.warning("rerank_chunks called with empty chunk list")
        return [], 1.0

    # No reranking needed if chunks already fit in top_n
    if len(chunks) <= top_n:
        logger.info(
            "Skipping rerank — only %d chunks, requested top_%d — "
            "returning as-is with score 1.0",
            len(chunks),
            top_n,
        )
        return chunks, 1.0

    client = _get_client()

    # Cohere not configured — return raw order, fail open on score
    if client is None:
        logger.info(
            "Reranker not available — returning top %d chunks in retrieval order "
            "with fallback score 1.0 (guardrail disabled)",
            top_n,
        )
        return chunks[:top_n], 1.0

    # ── Call Cohere Rerank API ────────────────────────────────────────────────
    try:
        response = client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=chunks,
            top_n=top_n,
        )

        reranked = [chunks[r.index] for r in response.results]

        # Top score — first result is highest scored (results sorted descending)
        top_score = (
            float(response.results[0].relevance_score) if response.results else 1.0
        )
        bottom_score = (
            float(response.results[-1].relevance_score) if response.results else 1.0
        )

        logger.info(
            "Cohere rerank complete — input: %d → output: %d | "
            "top_score: %.3f | bottom_score: %.3f",
            len(chunks),
            len(reranked),
            top_score,
            bottom_score,
        )

        return reranked, top_score

    except Exception as exc:
        # Cohere API failed — fail open, don't let reranker failure break chat
        logger.error(
            "Cohere rerank API failed: %s — "
            "returning raw order top %d with fallback score 1.0",
            exc,
            top_n,
        )
        return chunks[:top_n], 1.0
