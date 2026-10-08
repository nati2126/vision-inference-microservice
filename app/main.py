"""FastAPI application factory and lifespan."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.models.factory import create_backend
from app.schemas.detection import ErrorResponse
from app.services.detection_service import DetectionService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    setup_logging(log_level=settings.log_level, log_format=settings.log_format)
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        environment=settings.environment,
    )

    model = create_backend(
        settings.model_backend,
        model_path=settings.model_path,
        device=settings.model_device,
        confidence_threshold=settings.model_confidence_threshold,
        iou_threshold=settings.model_iou_threshold,
        max_detections=settings.model_max_detections,
        weights_dir=settings.model_weights_dir,
    )
    model.load()

    detection_service = DetectionService(model=model)

    app.state.settings = settings
    app.state.model = model
    app.state.detection_service = detection_service

    logger.info("application_ready")
    yield

    logger.info("application_shutting_down")
    model.unload()
    logger.info("application_stopped")


def register_exception_handlers(app: FastAPI) -> None:
    """Return every error in the ``ErrorResponse`` shape."""

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
        logger.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error="An unexpected error occurred.").model_dump(),
        )


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Object detection inference service for YOLOv8 with interchangeable "
            "PyTorch, ONNX Runtime, OpenVINO and TensorRT backends."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    allow_credentials = settings.cors_allow_credentials
    if allow_credentials and "*" in settings.cors_allow_origins:
        # Browsers reject credentialed responses with a wildcard origin.
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

    register_exception_handlers(app)

    app.include_router(v1_router)

    return app


app = create_app()
