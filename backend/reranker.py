import logging
import cohere
from config import get_settings

settings = get_settings()
logger = logging.getLogger("ragbot.reranker")

# ── Lazy client initialization ────────────────────────────────────────────────
# Client is only created if cohere_api_key is set in config.
# If the key is missing or empty, all rerank calls fall back gracefully
# to returning the top N raw chunks in their original hybrid-search order.
_cohere_client = None


def _get_client() -> cohere.Client | None:
    """
    Return a Cohere client, initializing it on first call.
    Returns None if no API key is configured — callers must handle this.
    """
    global _cohere_client
    if _cohere_client is None:
        if settings.cohere_api_key:
            _cohere_client = cohere.Client(api_key=settings.cohere_api_key)
            logger.info("Cohere reranker initialized — model: rerank-english-v3.0")
        else:
            logger.warning(
                "COHERE_API_KEY not set — reranker disabled. "
                "Retrieval will return raw hybrid-search order."
            )
    return _cohere_client


def rerank_chunks(query: str, chunks: list[str], top_n: int = None) -> list[str]:
    """
    Rerank a list of retrieved chunks by relevance to the query using
    Cohere's cross-encoder rerank model.

    What this does that embedding similarity can't:
    Cross-encoders read the query AND each chunk together as a pair,
    scoring how well they match. Embedding similarity only compares
    independent vectors — it doesn't understand the relationship between
    the two texts. Cohere's model scores that relationship directly.

    Args:
        query:   The user's question (original, not HyDE-expanded)
        chunks:  List of retrieved chunk strings from hybrid search
        top_n:   How many top chunks to return. Defaults to top_k_reranked
                 from config (5).

    Returns:
        List of top_n chunks sorted by Cohere relevance score, highest first.
        Falls back to chunks[:top_n] in original order if Cohere is
        unavailable or the API call fails.
    """
    if top_n is None:
        top_n = settings.top_k_reranked

    # Edge cases — nothing to rerank
    if not chunks:
        logger.warning("rerank_chunks called with empty chunk list")
        return []

    # If we have fewer or equal chunks than requested, no reranking needed
    if len(chunks) <= top_n:
        logger.info(
            "Skipping rerank — only %d chunks available, requested top_%d",
            len(chunks),
            top_n,
        )
        return chunks

    # Get client — may be None if key not configured
    client = _get_client()

    if client is None:
        logger.info(
            "Reranker not available — returning top %d chunks in retrieval order", top_n
        )
        return chunks[:top_n]

    # ── Call Cohere Rerank API ────────────────────────────────────────────────
    try:
        response = client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=chunks,
            top_n=top_n,
        )

        # response.results is sorted by relevance_score descending (best first)
        # r.index is the position of the chunk in the original chunks list
        reranked = [chunks[r.index] for r in response.results]

        logger.info(
            "Cohere rerank complete — input: %d chunks → output: %d chunks | "
            "top score: %.3f | bottom score: %.3f",
            len(chunks),
            len(reranked),
            response.results[0].relevance_score if response.results else 0,
            response.results[-1].relevance_score if response.results else 0,
        )

        return reranked

    except Exception as exc:
        # Never let a reranker failure break the chat pipeline.
        # Log it, fall back to raw order, keep serving the user.
        logger.error(
            "Cohere rerank API call failed: %s — "
            "falling back to raw retrieval order, returning top %d",
            exc,
            top_n,
        )
        return chunks[:top_n]
