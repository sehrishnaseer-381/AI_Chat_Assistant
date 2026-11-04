
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "AI Chat Assistant API"
    app_version: str = "0.1.0"
    environment: str = "development"

    api_prefix: str = "/api/v1"
    frontend_origin: str = "http://localhost:5173"

    groq_api_key: str | None = None
    groq_vision_model: str = Field(
        default="qwen/qwen3.8-27b",
        min_length=1,
    )
    groq_max_completion_tokens: int = Field(
        default=4096,
        ge=1,
        le=65536,
    )
    groq_default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    groq_temperature_override: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
    )
    database_url: str

    jwt_secret_key: str
    jwt_access_token_expire_minutes: int = 30

    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/google/callback"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
