
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "AI Chat Assistant API"
    app_version: str = "0.1.0"
    environment: str = "development"

    api_prefix: str = "/api/v1"
    frontend_origin: str = "http://localhost:5173"

    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str | None = None
    database_url: str

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
