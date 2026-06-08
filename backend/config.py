from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str
    ragbot_api_key: str = "dev-secret-key"
    groq_model: str = "llama3-8b-8192"
    google_credentials_path: str = "google-credentials.json"
    google_sheet_name: str = "RAG Chatbot Leads"
    email_sender: str
    email_password: str
    email_recipient: str
    max_file_size_mb: int = 10
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    cors_origin: str = "http://localhost:3000"
    env: str = "development"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
