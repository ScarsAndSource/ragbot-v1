"""
config.py — RAGbot v2
Phase 1: added Qdrant, Upstash, Supabase, Langfuse vars; Google creds from env
Phase 2: added qdrant_payload_schema_version
Phase 3: recalibrated thresholds; added cache_lru_eviction, cache_ttl_seconds, lead_classifier_threshold
Phase 4: groq_fallback_model, groq_streaming
Phase 5: replace email_sender/email_password with gmail_credentials_path
Phase 6: add client_domain, client_name, client_practice_info
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"   # P4: 429/connection fallback
    groq_streaming: bool = True                           # P4: enables /chat/stream

    # ── Reranker ──────────────────────────────────────────────────────────────
    cohere_api_key: str = ""

    # ── Auth ──────────────────────────────────────────────────────────────────
    ragbot_api_key: str = "dev-secret-key"

    # ── Google Sheets ─────────────────────────────────────────────────────────
    google_credentials_path: str = "/tmp/google-credentials.json"
    google_sheet_name: str = "RAG Chatbot Leads"

    # ── Email (SMTP — replaced with Gmail API in Phase 5) ─────────────────────
    email_sender: str = ""
    email_password: str = ""
    email_recipient: str = ""
    # Phase 5 — gmail_credentials_path: str = "gmail-credentials.json"

    # ── File handling ─────────────────────────────────────────────────────────
    max_file_size_mb: int = 10

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 400
    chunk_overlap: int = 60

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_results: int = 20
    top_k_reranked: int = 5
    use_hyde: bool = False

    # ── Semantic cache ────────────────────────────────────────────────────────
    cache_similarity_threshold: float = 0.88
    cache_max_size: int = 100
    cache_lru_eviction: bool = True
    cache_ttl_seconds: int = 86400

    # ── Output guardrail ──────────────────────────────────────────────────────
    guardrail_min_rerank_score: float = 0.18

    # ── Lead classifier ───────────────────────────────────────────────────────
    lead_classifier_threshold: float = 0.6

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_prefix: str = "ragbot"
    qdrant_payload_schema_version: int = 2

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