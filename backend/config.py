from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Reranker ──────────────────────────────────────────────────────────────
    cohere_api_key: str = ""  # leave empty to disable reranker gracefully

    # ── Auth ──────────────────────────────────────────────────────────────────
    ragbot_api_key: str = "dev-secret-key"

    # ── Google Sheets + Email ─────────────────────────────────────────────────
    google_credentials_path: str = "google-credentials.json"
    google_sheet_name: str = "RAG Chatbot Leads"
    email_sender: str
    email_password: str
    email_recipient: str

    # ── File handling ─────────────────────────────────────────────────────────
    max_file_size_mb: int = 10

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 400
    chunk_overlap: int = 60

    # ── Retrieval pipeline ────────────────────────────────────────────────────
    # Phase 1 had this at 6 — now that the reranker exists to narrow it back
    # down, we retrieve wide (20) and let Cohere pick the best 5 for the LLM.
    # This is the target architecture described in the upgrade doc.
    top_k_results: int = 20  # how many chunks hybrid search retrieves
    top_k_reranked: int = 5  # how many go to the LLM after reranking

    # ── CORS + env ────────────────────────────────────────────────────────────
    cors_origin: str = "http://localhost:3000"
    env: str = "development"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
