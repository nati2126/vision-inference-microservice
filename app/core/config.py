"""Settings from environment variables or .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "vision-inference-microservice"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    model_backend: Literal["pytorch", "onnxruntime", "openvino", "tensorrt"] = "pytorch"
    # MODEL_NAME is accepted as a legacy alias.
    model_path: str = Field(
        default="yolov8n.pt", validation_alias=AliasChoices("model_path", "model_name")
    )
    model_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    model_iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    model_max_detections: int = Field(default=300, gt=0)
    model_device: Literal["cpu", "cuda", "mps"] = "cpu"
    # Where bare .pt filenames are downloaded; the container's WORKDIR is read-only.
    model_weights_dir: Path = Path.home() / ".cache" / "vision-inference" / "weights"

    # JSON list, e.g. CORS_ALLOW_ORIGINS='["https://app.example.com"]'.
    cors_allow_origins: list[str] = ["*"]
    cors_allow_credentials: bool = False

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
