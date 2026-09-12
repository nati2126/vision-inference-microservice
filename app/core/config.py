"""Application settings powered by pydantic-settings.

All configuration is driven by environment variables (or a .env file),
following the 12-Factor App methodology. Defaults are tuned for local
development; production values should be injected via Docker env / secrets.
"""

from functools import lru_cache
from pathlib import Path
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
    # Directory that bare weight filenames are resolved against. Ultralytics
    # downloads a relative filename into the *current working directory*, which
    # is not writable in the container (WORKDIR is root-owned, the process runs
    # as an unprivileged user). Handing it an absolute path under a writable,
    # volume-mounted directory fixes that and makes the weights cache persist.
    model_weights_dir: Path = Path.home() / ".cache" / "vision-inference" / "weights"

    # ── CORS ─────────────────────────────────────────────────
    # Set as a JSON list, e.g. CORS_ALLOW_ORIGINS='["https://app.example.com"]'.
    # ``["*"]`` is a development default — narrow it in production.
    cors_allow_origins: list[str] = ["*"]
    # Credentialed requests cannot be combined with a wildcard origin: the
    # browser rejects ``Access-Control-Allow-Origin: *`` whenever credentials
    # are included, so this defaults off and must be enabled alongside an
    # explicit origin list.
    cors_allow_credentials: bool = False

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
