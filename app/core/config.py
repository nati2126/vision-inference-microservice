"""Application settings powered by pydantic-settings.

All configuration is driven by environment variables (or a .env file),
following the 12-Factor App methodology. Defaults are tuned for local
development; production values should be injected via Docker env / secrets.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────
    app_name: str = "vision-inference-microservice"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # ── Server ───────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Model ────────────────────────────────────────────────
    model_name: str = "yolov8n.pt"
    model_confidence_threshold: float = 0.25
    model_device: str = "cpu"  # "cpu" | "cuda" | "mps"

    # ── Logging ──────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, singleton Settings instance.

    Using ``lru_cache`` guarantees .env is read only once and the same
    validated object is reused across the application lifetime.
    """
    return Settings()
