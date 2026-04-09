"""
Application configuration – loaded from environment variables / .env file.
Uses pydantic-settings for validation and type coercion.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Centralised, validated app settings."""

    # ── Application ──────────────────────────────────────────
    APP_NAME: str = "Lucida Lead Scoring API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # ── Database (Turso / libSQL) ────────────────────────────
    TURSO_DATABASE_URL: str = ""
    TURSO_AUTH_TOKEN: str = ""

    # ── Authentication (Clerk) ──────────────────────────────
    CLERK_SECRET_KEY: str = ""
    CLERK_PUBLISHABLE_KEY: str = ""

    # ── CORS ─────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://localhost:8000"

    # ── ML ───────────────────────────────────────────────────
    MODEL_ARTIFACTS_DIR: str = "./model_artifacts"
    MAX_CSV_SIZE_MB: int = 200

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
