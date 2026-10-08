"""GET /health."""

from fastapi import APIRouter, Request

from app.models.base import InferenceBackend
from app.schemas.detection import HealthResponse

router = APIRouter()


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
    )
