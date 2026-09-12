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

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.models.yolo_model import YOLOModel
from app.schemas.detection import ErrorResponse
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
        weights_dir=settings.model_weights_dir,
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


def register_exception_handlers(app: FastAPI) -> None:
    """Make error responses match the documented ``ErrorResponse`` schema.

    FastAPI's defaults return ``{"detail": ...}``, while the endpoints
    advertise ``ErrorResponse`` (``{"error": ..., "detail": ...}``) for their
    4xx / 5xx responses. Without these handlers the OpenAPI contract and the
    actual payloads disagree.
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=str(exc.detail)).model_dump(),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,  # name differs across starlette versions
            content=ErrorResponse(
                error="Request validation failed.",
                detail=str(exc.errors()),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals to the caller; the traceback goes to the logs.
        logger.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error="An unexpected error occurred.").model_dump(),
        )


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
    allow_credentials = settings.cors_allow_credentials
    if allow_credentials and "*" in settings.cors_allow_origins:
        # Browsers refuse a credentialed response carrying the wildcard
        # origin, so this combination silently breaks every such request.
        logger.warning(
            "cors_credentials_disabled_for_wildcard_origin",
            hint="Set CORS_ALLOW_ORIGINS to an explicit list to use credentials.",
        )
        allow_credentials = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Error handling ───────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ──────────────────────────────────────────────
    app.include_router(v1_router)

    return app


# Uvicorn entry point
app = create_app()
