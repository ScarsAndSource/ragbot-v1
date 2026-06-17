"""
config.py — RAGbot v2, Phase 1

Changes from v1:
  [P1-C1] ADDED: google_credentials_json — full service account JSON as env var
                  Render's ephemeral filesystem resets on every deploy; this
                  lets main.py write creds to a temp file at startup instead
                  of relying on a file that won't survive a restart.
  [P1-C2] ADDED: qdrant_url, qdrant_api_key, qdrant_collection_prefix
                  Qdrant Cloud free cluster. Phase 2 migrates all storage here.
  [P1-C3] ADDED: upstash_redis_url, upstash_redis_token
                  Upstash REST-based Redis. Session store uses this in Phase 1.
                  Phase 3 replaces the in-memory SemanticCache with it.
  [P1-C4] ADDED: supabase_url, supabase_key
                  Phase 2 uses Supabase for document dedup table.
                  Phase 5 uses it for pipeline_logs and low_confidence_leads.
  [P1-C5] ADDED: langfuse_public_key, langfuse_secret_key, langfuse_host
                  Phase 1 sends a startup trace. Phase 5 wires per-stage spans.
  [P1-C6] KEPT:  All v1 vars unchanged. Nothing removed in Phase 1.
                  SMTP vars stay until Phase 5 replaces them with Gmail API.

Phases that will touch this file next:
  Phase 2: qdrant_payload_schema_version constant added
  Phase 3: cache_similarity_threshold recalibrated to 0.88,
            guardrail_min_rerank_score recalibrated to 0.18,
            cache_lru_eviction and cache_ttl_seconds added
  Phase 4: groq_fallback_model and groq_streaming added
  Phase 5: email_sender / email_password removed (SMTP gone),
            lead_confidence_threshold added,
            gmail_credentials_path added
  Phase 6: client_domain, client_name, client_practice_info added
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    # groq_fallback_model added in Phase 4
    # groq_streaming added in Phase 4

    # ── Reranker ──────────────────────────────────────────────────────────────
    cohere_api_key: str = ""

    # ── Auth ──────────────────────────────────────────────────────────────────
    ragbot_api_key: str = "dev-secret-key"

    # ── Google Sheets ─────────────────────────────────────────────────────────
    # [P1-C1] google_credentials_json: full service account JSON string.
    # Paste the entire contents of your service account .json file as this
    # env var on Render. main.py writes it to a temp file at startup.
    # If empty, falls back to google_credentials_path for local dev.
    google_credentials_json: str = ""
    google_credentials_path: str = "google-credentials.json"
    google_sheet_name: str = "RAG Chatbot Leads"

    # ── Email (SMTP — replaced with Gmail API in Phase 5) ─────────────────────
    # [P1-C6] These stay until Phase 5. Do NOT add new SMTP-dependent code.
    email_sender: str = ""
    email_password: str = ""
    email_recipient: str = ""

    # ── File handling ─────────────────────────────────────────────────────────
    max_file_size_mb: int = 10

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Used by chat.py. rag.py uses its own internal constants in Phase 2.
    chunk_size: int = 400
    chunk_overlap: int = 60

    # ── Retrieval pipeline ────────────────────────────────────────────────────
    top_k_results: int = 20
    top_k_reranked: int = 5
    use_hyde: bool = False

    # ── Semantic cache ────────────────────────────────────────────────────────
    # Threshold recalibrated to 0.88 in Phase 3 after eval run.
    # cache_lru_eviction and cache_ttl_seconds added in Phase 3.
    cache_similarity_threshold: float = 0.92
    cache_max_size: int = 100

    # ── Output guardrail ──────────────────────────────────────────────────────
    # Threshold recalibrated to 0.18 in Phase 3 after eval run.
    guardrail_min_rerank_score: float = 0.20

    # ── CORS + env ────────────────────────────────────────────────────────────
    cors_origin: str = "http://localhost:3000"
    env: str = "development"

    # ── [P1-C2] Qdrant ────────────────────────────────────────────────────────
    # Free cluster URL from cloud.qdrant.io — looks like:
    # https://xxxx-xxxx-xxxx.us-east4-0.gcp.cloud.qdrant.io:6333
    # qdrant_collection_prefix: collections named {prefix}_{session_id}
    # Phase 2 does all collection creation/querying.
    # Phase 1 only calls ping_qdrant() to verify the cluster is reachable.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_prefix: str = "ragbot"
    # qdrant_payload_schema_version added in Phase 2

    # ── [P1-C3] Upstash Redis ─────────────────────────────────────────────────
    # Free tier at upstash.com. REST-based — no open TCP socket needed.
    # upstash_redis_url looks like: https://xxxx.upstash.io
    # upstash_redis_token is the token shown in the Upstash console.
    # Phase 1: session store prefers this over legacy REDIS_URL.
    # Phase 3: SemanticCache fully replaced with Upstash-backed version.
    upstash_redis_url: str = ""
    upstash_redis_token: str = ""

    # ── [P1-C4] Supabase ──────────────────────────────────────────────────────
    # Free tier at supabase.com. Project URL and anon/service key.
    # Phase 2: documents dedup table (document_exists, register_document).
    # Phase 5: pipeline_logs table, low_confidence_leads table.
    supabase_url: str = ""
    supabase_key: str = ""

    # ── [P1-C5] Langfuse ──────────────────────────────────────────────────────
    # Free tier at cloud.langfuse.com (or self-hosted).
    # Phase 1: startup trace sent so dashboard is visible immediately.
    # Phase 5: per-stage spans wired inside get_chat_response.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()