import logging
import numpy as np
from rag import embedding_model
from config import get_settings

settings = get_settings()
logger = logging.getLogger("ragbot.cache")


class SemanticCache:
    """
    In-memory semantic similarity cache for RAG query-answer pairs.

    How it works:
        Every answered query gets stored as (embedding, answer, metadata).
        On a new query, we embed it and compute cosine similarity against
        every stored embedding for that session. If the highest similarity
        exceeds the threshold, we return the cached answer immediately —
        skipping hybrid search, Cohere rerank, and the Groq LLM call.

    Why per-session scoping:
        Each session has its own document. Client A's chatbot cached answers
        about their return policy must never surface when Client B asks about
        their product manual. Session IDs are the isolation boundary.

    Why FIFO eviction:
        Oldest entries are least likely to be re-asked. Simplest eviction
        strategy that keeps memory bounded without LRU overhead.

    Thread safety:
        This implementation is single-process safe. FastAPI's async routes
        don't create true threads for synchronous code, so the in-memory
        dict is safe on Render's single-instance deployment. For multi-instance
        deployments, replace _store with Redis — interface stays identical.

    Redis upgrade path (when you need it):
        Replace self._store dict operations with:
        - redis.get(f"cache:{session_id}") → json.loads → list of entries
        - redis.setex(f"cache:{session_id}", ttl, json.dumps(entries))
        Embeddings stored as base64-encoded numpy arrays in the JSON.
        Everything else stays the same.
    """

    def __init__(self):
        # session_id → list of cache entry dicts
        # Each entry: {embedding, answer, query, lead_triggered, chunks}
        self._store: dict[str, list[dict]] = {}

    # ── Similarity ────────────────────────────────────────────────────────────
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Standard cosine similarity between two embedding vectors.
        Returns 0.0 if either vector is zero-norm (degenerate case).
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ── Read ──────────────────────────────────────────────────────────────────
    def get(
        self,
        query: str,
        session_id: str,
    ) -> tuple[str, bool, list[str]] | None:
        """
        Check if a semantically similar query has already been answered.

        Scans all cache entries for this session, finds the one with the
        highest cosine similarity to the incoming query, and returns its
        stored answer if similarity >= threshold.

        Returns:
            (answer, lead_triggered, chunks) if cache hit
            None if cache miss
        """
        entries = self._store.get(session_id, [])

        if not entries:
            logger.debug("Cache empty for session=%s — miss", session_id)
            return None

        # Embed the incoming query once
        query_emb = embedding_model.encode([query])[0]

        best_sim = 0.0
        best_entry = None

        for entry in entries:
            sim = self._cosine_similarity(query_emb, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= settings.cache_similarity_threshold and best_entry is not None:
            logger.info(
                "Cache HIT — session=%s | similarity=%.4f | threshold=%.2f | "
                "matched: '%s...' | saved: hybrid search + rerank + LLM call",
                session_id,
                best_sim,
                settings.cache_similarity_threshold,
                best_entry["query"][:60],
            )
            return (
                best_entry["answer"],
                best_entry["lead_triggered"],
                best_entry["chunks"],
            )

        logger.info(
            "Cache MISS — session=%s | best similarity=%.4f | threshold=%.2f | "
            "proceeding to full pipeline",
            session_id,
            best_sim,
            settings.cache_similarity_threshold,
        )
        return None

    # ── Write ─────────────────────────────────────────────────────────────────
    def set(
        self,
        query: str,
        answer: str,
        session_id: str,
        lead_triggered: bool = False,
        chunks: list[str] | None = None,
    ) -> None:
        """
        Store a query-answer pair after a successful full pipeline run.

        What gets cached:
            Normal document Q&A responses — yes.
            "I don't have that information" responses — yes.
                Within a session the document is fixed, so negative answers
                are correct and should be cached too.

        What does NOT get cached:
            Lead-triggered responses — no.
                When a user is in purchase/contact intent, we want the
                pipeline to run fresh each time. Lead detection may behave
                differently across messages and caching these breaks the
                lead capture flow.

            Empty or too-short answers — no.
                These are error states, not real answers.
        """
        if chunks is None:
            chunks = []

        # Skip caching lead-triggered responses
        if lead_triggered:
            logger.debug(
                "Cache SKIP (lead triggered) — session=%s | query: '%s...'",
                session_id,
                query[:60],
            )
            return

        # Skip caching empty or error responses
        if not answer or len(answer.strip()) < 10:
            logger.debug("Cache SKIP (empty/short answer) — session=%s", session_id)
            return

        # Embed query for storage
        query_emb = embedding_model.encode([query])[0]

        if session_id not in self._store:
            self._store[session_id] = []

        self._store[session_id].append(
            {
                "embedding": query_emb,
                "answer": answer,
                "query": query,
                "lead_triggered": lead_triggered,
                "chunks": chunks,
            }
        )

        current_size = len(self._store[session_id])

        # FIFO eviction — remove oldest entry when over limit
        if current_size > settings.cache_max_size:
            evicted = self._store[session_id].pop(0)
            logger.debug(
                "Cache EVICT (FIFO) — session=%s | evicted: '%s...'",
                session_id,
                evicted["query"][:40],
            )
            current_size -= 1

        logger.info(
            "Cache STORE — session=%s | entries now: %d/%d | query: '%s...'",
            session_id,
            current_size,
            settings.cache_max_size,
            query[:60],
        )

    # ── Clear ─────────────────────────────────────────────────────────────────
    def clear(self, session_id: str) -> int:
        """
        Remove all cache entries for a session.
        Called when a session's conversation history is cleared.
        Returns number of entries removed.
        """
        entries = self._store.pop(session_id, [])
        count = len(entries)
        if count > 0:
            logger.info(
                "Cache CLEAR — session=%s | removed %d entries", session_id, count
            )
        return count

    # ── Stats ─────────────────────────────────────────────────────────────────
    def stats(self, session_id: str) -> dict:
        """
        Return cache statistics for a session.
        Used for logging and the future analytics dashboard.
        """
        entries = self._store.get(session_id, [])
        return {
            "session_id": session_id,
            "entries": len(entries),
            "max_entries": settings.cache_max_size,
            "threshold": settings.cache_similarity_threshold,
            "cached_queries": [e["query"][:60] for e in entries],
        }

    # ── Global stats ──────────────────────────────────────────────────────────
    def global_stats(self) -> dict:
        """
        Return stats across all sessions.
        Useful for understanding overall cache utilization.
        """
        total_entries = sum(len(v) for v in self._store.values())
        return {
            "total_sessions_cached": len(self._store),
            "total_entries": total_entries,
            "sessions": list(self._store.keys()),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
# One cache instance for the entire server process.
# All imports of this module get the same object.
semantic_cache = SemanticCache()
