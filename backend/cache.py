"""
cache.py — RAGbot v2, Phase 3
Full replacement of in-memory SemanticCache with Upstash Redis backend.

External interface is IDENTICAL to v1 — chat.py import unchanged:
    from cache import semantic_cache
    semantic_cache.get(query, session_id)
    semantic_cache.set(query, answer, session_id, lead_triggered, chunks)
    semantic_cache.clear(session_id)
    semantic_cache.stats(session_id)
    semantic_cache.global_stats()

Storage layout (Redis):
    HASH  cache:{session_id}:entries   field=query_hash → JSON payload
    ZSET  cache:{session_id}:lru       member=query_hash, score=unix_ts

Embedding storage:
    float32 numpy array → tobytes() → base64 string
    1024 dims × 4 bytes = 4 KB → ~5.5 KB base64 per entry
    100 entries ≈ 550 KB — acceptable for Upstash REST calls

LRU eviction:
    ZSet score = last-access timestamp.
    On every cache miss / set: if len > cache_max_size, pop lowest score.
    On hit: zadd with XX to refresh score without creating duplicate.

TTL:
    Both keys get EXPIRE set on every write = cache_ttl_seconds (24 h default).
    Effectively a sliding window — active sessions stay alive.

Fallback:
    When Upstash not configured or unreachable, silently falls back to
    the same in-memory dict used in v1. Zero disruption to the pipeline.
"""

import base64
import hashlib
import json
import logging
import time
from typing import Optional

import numpy as np

from config import get_settings
from rag import _embed_query

settings = get_settings()
logger = logging.getLogger("ragbot.cache")

# ── Upstash Redis lazy client ─────────────────────────────────────────────────
_redis = None


def _get_redis():
    global _redis
    if _redis is None:
        if settings.upstash_redis_url and settings.upstash_redis_token:
            try:
                from upstash_redis import Redis
                _redis = Redis(
                    url=settings.upstash_redis_url,
                    token=settings.upstash_redis_token,
                )
                _redis.ping()
                logger.info("Upstash Redis cache connected")
            except Exception as exc:
                logger.warning(
                    "Upstash Redis unavailable (%s) — falling back to in-memory cache",
                    exc,
                )
                _redis = None
        else:
            logger.info("Upstash not configured — using in-memory cache")
    return _redis


# ── Embedding serialization ───────────────────────────────────────────────────
def _emb_encode(emb: list[float]) -> str:
    """float list → float32 bytes → base64 string. ~5.5 KB for 1024 dims."""
    return base64.b64encode(
        np.array(emb, dtype=np.float32).tobytes()
    ).decode("ascii")


def _emb_decode(b64: str) -> np.ndarray:
    """base64 string → float32 numpy array."""
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32)


# ── Cosine similarity ─────────────────────────────────────────────────────────
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Key helpers ───────────────────────────────────────────────────────────────
def _entries_key(session_id: str) -> str:
    return f"cache:{session_id}:entries"


def _lru_key(session_id: str) -> str:
    return f"cache:{session_id}:lru"


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# SemanticCache — Redis path + in-memory fallback
# ══════════════════════════════════════════════════════════════════════════════
class SemanticCache:
    """
    Semantic similarity cache backed by Upstash Redis.
    Falls back to in-memory dict when Redis is unavailable.

    All public methods have identical signatures to v1 — zero changes in
    chat.py or main.py are needed.
    """

    def __init__(self):
        # In-memory fallback store (same shape as v1)
        self._mem: dict[str, list[dict]] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def get(
        self,
        query: str,
        session_id: str,
    ) -> tuple[str, bool, list[str]] | None:
        r = _get_redis()
        if r is not None:
            return self._redis_get(query, session_id, r)
        return self._mem_get(query, session_id)

    def set(
        self,
        query: str,
        answer: str,
        session_id: str,
        lead_triggered: bool = False,
        chunks: list[str] | None = None,
    ) -> None:
        if chunks is None:
            chunks = []

        # Never cache lead-triggered or empty answers — same rule as v1
        if lead_triggered:
            logger.debug("Cache SKIP (lead) — session=%s", session_id)
            return
        if not answer or len(answer.strip()) < 10:
            logger.debug("Cache SKIP (short answer) — session=%s", session_id)
            return

        r = _get_redis()
        if r is not None:
            self._redis_set(query, answer, session_id, lead_triggered, chunks, r)
        else:
            self._mem_set(query, answer, session_id, lead_triggered, chunks)

    def clear(self, session_id: str) -> int:
        r = _get_redis()
        if r is not None:
            return self._redis_clear(session_id, r)
        return self._mem_clear(session_id)

    def stats(self, session_id: str) -> dict:
        r = _get_redis()
        if r is not None:
            return self._redis_stats(session_id, r)
        return self._mem_stats(session_id)

    def global_stats(self) -> dict:
        r = _get_redis()
        if r is not None:
            return {"backend": "upstash_redis", "note": "per-session stats via stats()"}
        total = sum(len(v) for v in self._mem.values())
        return {
            "backend": "in_memory",
            "total_sessions_cached": len(self._mem),
            "total_entries": total,
            "sessions": list(self._mem.keys()),
        }

    # ── Redis implementation ──────────────────────────────────────────────────

    def _redis_get(
        self, query: str, session_id: str, r
    ) -> tuple[str, bool, list[str]] | None:
        ek = _entries_key(session_id)
        lk = _lru_key(session_id)

        try:
            raw_entries = r.hgetall(ek)
        except Exception as exc:
            logger.warning("Redis HGETALL failed: %s — cache miss", exc)
            return None

        if not raw_entries:
            logger.debug("Cache empty (Redis) — session=%s", session_id)
            return None

        query_emb = np.array(_embed_query(query), dtype=np.float32)
        best_sim = 0.0
        best_entry = None
        best_hash = None

        for qhash, payload_str in raw_entries.items():
            try:
                entry = json.loads(payload_str)
                stored_emb = _emb_decode(entry["embedding_b64"])
                sim = _cosine(query_emb, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
                    best_hash = qhash
            except Exception as exc:
                logger.debug("Skipping malformed cache entry %s: %s", qhash, exc)
                continue

        if best_sim >= settings.cache_similarity_threshold and best_entry:
            # Refresh LRU score on hit
            try:
                r.zadd(lk, {best_hash: time.time()})
            except Exception:
                pass

            logger.info(
                "Cache HIT (Redis) — session=%s | sim=%.4f | matched: '%s…'",
                session_id, best_sim, best_entry.get("query", "")[:60],
            )
            return (
                best_entry["answer"],
                best_entry["lead_triggered"],
                best_entry.get("chunks", []),
            )

        logger.info(
            "Cache MISS (Redis) — session=%s | best_sim=%.4f | threshold=%.2f",
            session_id, best_sim, settings.cache_similarity_threshold,
        )
        return None

    def _redis_set(
        self,
        query: str,
        answer: str,
        session_id: str,
        lead_triggered: bool,
        chunks: list[str],
        r,
    ) -> None:
        ek = _entries_key(session_id)
        lk = _lru_key(session_id)
        qhash = _query_hash(query)
        now = time.time()

        query_emb = _embed_query(query)
        entry = {
            "embedding_b64": _emb_encode(query_emb),
            "answer": answer,
            "query": query,
            "lead_triggered": lead_triggered,
            "chunks": chunks,
            "timestamp": now,
        }

        try:
            r.hset(ek, qhash, json.dumps(entry))
            r.zadd(lk, {qhash: now})

            # LRU eviction — remove oldest if over limit
            if settings.cache_lru_eviction:
                count = r.zcard(lk)
                if count > settings.cache_max_size:
                    oldest = r.zrange(lk, 0, 0)
                    if oldest:
                        evict_hash = oldest[0]
                        r.hdel(ek, evict_hash)
                        r.zrem(lk, evict_hash)
                        logger.debug(
                            "Cache EVICT (LRU/Redis) — session=%s | removed hash=%s",
                            session_id, evict_hash,
                        )
                        count -= 1

            # Sliding TTL — keep active sessions alive
            r.expire(ek, settings.cache_ttl_seconds)
            r.expire(lk, settings.cache_ttl_seconds)

            current = r.hlen(ek)
            logger.info(
                "Cache STORE (Redis) — session=%s | entries=%d | query: '%s…'",
                session_id, current, query[:60],
            )
        except Exception as exc:
            logger.error("Redis cache SET failed: %s — falling back to mem", exc)
            self._mem_set(query, answer, session_id, lead_triggered, chunks)

    def _redis_clear(self, session_id: str, r) -> int:
        ek = _entries_key(session_id)
        lk = _lru_key(session_id)
        try:
            count = r.hlen(ek) or 0
            r.delete(ek)
            r.delete(lk)
            if count:
                logger.info(
                    "Cache CLEAR (Redis) — session=%s | removed %d entries",
                    session_id, count,
                )
            return count
        except Exception as exc:
            logger.error("Redis cache CLEAR failed: %s", exc)
            return 0

    def _redis_stats(self, session_id: str, r) -> dict:
        try:
            count = r.hlen(_entries_key(session_id)) or 0
        except Exception:
            count = 0
        return {
            "backend": "upstash_redis",
            "session_id": session_id,
            "entries": count,
            "max_entries": settings.cache_max_size,
            "threshold": settings.cache_similarity_threshold,
            "ttl_seconds": settings.cache_ttl_seconds,
        }

    # ── In-memory fallback ────────────────────────────────────────────────────

    def _mem_get(
        self, query: str, session_id: str
    ) -> tuple[str, bool, list[str]] | None:
        entries = self._mem.get(session_id, [])
        if not entries:
            return None

        query_emb = np.array(_embed_query(query), dtype=np.float32)
        best_sim, best_entry = 0.0, None

        for e in entries:
            sim = _cosine(query_emb, e["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = e

        if best_sim >= settings.cache_similarity_threshold and best_entry:
            logger.info(
                "Cache HIT (mem) — session=%s | sim=%.4f | '%s…'",
                session_id, best_sim, best_entry["query"][:60],
            )
            return best_entry["answer"], best_entry["lead_triggered"], best_entry["chunks"]

        logger.info(
            "Cache MISS (mem) — session=%s | best_sim=%.4f", session_id, best_sim
        )
        return None

    def _mem_set(
        self,
        query: str,
        answer: str,
        session_id: str,
        lead_triggered: bool,
        chunks: list[str],
    ) -> None:
        query_emb = np.array(_embed_query(query), dtype=np.float32)

        if session_id not in self._mem:
            self._mem[session_id] = []

        self._mem[session_id].append({
            "embedding": query_emb,
            "answer": answer,
            "query": query,
            "lead_triggered": lead_triggered,
            "chunks": chunks,
        })

        if len(self._mem[session_id]) > settings.cache_max_size:
            evicted = self._mem[session_id].pop(0)
            logger.debug("Cache EVICT (FIFO/mem) — '%s…'", evicted["query"][:40])

        logger.info(
            "Cache STORE (mem) — session=%s | entries=%d | '%s…'",
            session_id, len(self._mem[session_id]), query[:60],
        )

    def _mem_clear(self, session_id: str) -> int:
        entries = self._mem.pop(session_id, [])
        count = len(entries)
        if count:
            logger.info("Cache CLEAR (mem) — session=%s | removed %d", session_id, count)
        return count

    def _mem_stats(self, session_id: str) -> dict:
        entries = self._mem.get(session_id, [])
        return {
            "backend": "in_memory",
            "session_id": session_id,
            "entries": len(entries),
            "max_entries": settings.cache_max_size,
            "threshold": settings.cache_similarity_threshold,
            "cached_queries": [e["query"][:60] for e in entries],
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
semantic_cache = SemanticCache()    