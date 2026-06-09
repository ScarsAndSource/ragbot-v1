from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Reranker ──────────────────────────────────────────────────────────────
    cohere_api_key: str = ""

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
    top_k_results: int = 20
    top_k_reranked: int = 5
    use_hyde: bool = False

    # ── Semantic cache ────────────────────────────────────────────────────────
    cache_similarity_threshold: float = 0.92
    cache_max_size: int = 100

    # ── CORS + env ────────────────────────────────────────────────────────────
    cors_origin: str = "http://localhost:3000"
    env: str = "development"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
