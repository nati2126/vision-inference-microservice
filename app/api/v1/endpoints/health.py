"""GET /health, plus the unversioned /livez and /readyz probes."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.models.base import InferenceBackend
from app.schemas.detection import HealthResponse

router = APIRouter()
probe_router = APIRouter(tags=["Probes"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns service status, version, model readiness and the active backend.",
)
async def health_check(request: Request) -> HealthResponse:
    model: InferenceBackend = request.app.state.model
    settings = request.app.state.settings

    return HealthResponse(
        status="healthy" if model.is_loaded else "degraded",
        version=settings.app_version,
        model_loaded=model.is_loaded,
        backend=model.name,
        precision=model.precision,
        device=model.device,
        max_batch_size=request.app.state.detection_service.max_batch_size,
    )


@probe_router.get("/livez", summary="Liveness probe: the process is serving requests")
async def livez() -> dict[str, str]:
    return {"status": "ok"}


@probe_router.get(
    "/readyz",
    summary="Readiness probe: the model is loaded",
    responses={503: {"description": "Model not loaded"}},
)
async def readyz(request: Request) -> JSONResponse:
    model: InferenceBackend | None = getattr(request.app.state, "model", None)
    if model is None or not model.is_loaded:
        return JSONResponse(
            {"status": "not_ready"}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    return JSONResponse({"status": "ready"})
