"""
config.py — RAGbot v2
Phase 1: added Qdrant, Upstash, Supabase, Langfuse vars; Google creds from env
Phase 2: added qdrant_payload_schema_version
Phase 3: recalibrated thresholds; added cache_lru_eviction, cache_ttl_seconds, lead_classifier_threshold
Phase 4: add groq_fallback_model, groq_streaming
Phase 5: replace email_sender/email_password with gmail_credentials_path
Phase 6: add client_domain, client_name, client_practice_info
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    # Phase 4 — groq_fallback_model: str = "llama-3.1-8b-instant"
    # Phase 4 — groq_streaming: bool = True

    # ── Reranker ──────────────────────────────────────────────────────────────
    cohere_api_key: str = ""

    # ── Auth ──────────────────────────────────────────────────────────────────
    ragbot_api_key: str = "dev-secret-key"

    # ── Google Sheets ─────────────────────────────────────────────────────────
    # P1: google_credentials_path now written from GOOGLE_CREDENTIALS_JSON env var
    #     at startup in main.py; keep field so gspread integration still reads it
    google_credentials_path: str = "/tmp/google-credentials.json"
    google_sheet_name: str = "RAG Chatbot Leads"

    # ── Email (SMTP — replaced with Gmail API in Phase 5) ─────────────────────
    email_sender: str = ""
    email_password: str = ""
    email_recipient: str = ""
    # Phase 5 — gmail_credentials_path: str = "gmail-credentials.json"

    # ── File handling ─────────────────────────────────────────────────────────
    max_file_size_mb: int = 10

    # ── Chunking (Phase 2 sizes wired in rag.py constants, not config) ────────
    chunk_size: int = 400  # kept for backward compat — not used by P2 chunker
    chunk_overlap: int = 60  # kept for backward compat

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_results: int = 20  # child chunks fetched per dense/sparse prefetch
    top_k_reranked: int = 5  # parents returned to LLM after Cohere rerank
    use_hyde: bool = False

    # ── Semantic cache ────────────────────────────────────────────────────────
    cache_similarity_threshold: float = 0.88  # P3: recalibrated from 0.92
    cache_max_size: int = 100
    cache_lru_eviction: bool = True  # P3: LRU eviction in Redis ZSet
    cache_ttl_seconds: int = 86400  # P3: Redis key TTL (24 h)

    # ── Output guardrail ──────────────────────────────────────────────────────
    guardrail_min_rerank_score: float = 0.18  # P3: recalibrated from 0.20

    # ── Lead classifier ───────────────────────────────────────────────────────
    lead_classifier_threshold: float = 0.6  # P3: min confidence to flag a lead

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_prefix: str = "ragbot"
    qdrant_payload_schema_version: int = 2  # P2: bump when payload schema changes

    # ── Upstash Redis ─────────────────────────────────────────────────────────
    upstash_redis_url: str = ""
    upstash_redis_token: str = ""

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = ""
    supabase_key: str = ""

    # ── Langfuse ──────────────────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # ── CORS + env ────────────────────────────────────────────────────────────
    cors_origin: str = "http://localhost:3000"
    env: str = "development"
    # Phase 6 — client_domain: str = ""
    # Phase 6 — client_name: str = "Demo Practice"
    # Phase 6 — client_practice_info: str = ""

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
