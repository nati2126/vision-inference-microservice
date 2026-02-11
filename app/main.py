"""Application factory — creates and configures the FastAPI app.

Uses the modern ``lifespan`` context manager pattern (FastAPI ≥ 0.95)
instead of the deprecated ``on_event("startup")`` / ``on_event("shutdown")``
decorators. The lifespan manager is responsible for:

1. Loading configuration
2. Initialising structured logging
3. Loading the ML model into memory (once)
4. Creating service instances
5. Tearing everything down gracefully on shutdown
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.models.yolo_model import YOLOModel
from app.services.detection_service import DetectionService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Everything before ``yield`` runs at startup; everything after runs at
    shutdown. Objects attached to ``app.state`` are available to all
    request handlers via ``request.app.state``.
    """
    settings = get_settings()

    # ── Logging ──────────────────────────────────────────────
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        environment=settings.environment,
    )

    # ── Model ────────────────────────────────────────────────
    model = YOLOModel(
        model_name=settings.model_name,
        device=settings.model_device,
        confidence_threshold=settings.model_confidence_threshold,
    )
    model.load()

    # ── Services ─────────────────────────────────────────────
    detection_service = DetectionService(model=model)

    # ── Attach to app state (dependency injection) ───────────
    app.state.settings = settings
    app.state.model = model
    app.state.detection_service = detection_service

    logger.info("application_ready")
    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("application_shutting_down")
    model.unload()
    logger.info("application_stopped")


def create_app() -> FastAPI:
    """Application factory — returns a fully configured FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade object detection inference microservice "
            "powered by YOLOv8."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────
    app.include_router(v1_router)

    return app


# Uvicorn entry point
app = create_app()
