"""Health-check endpoint.

Used by Docker HEALTHCHECK, Kubernetes liveness / readiness probes, and
load balancers to determine if the service is operational.
"""

from fastapi import APIRouter, Request

from app.schemas.detection import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns service status, version, and model readiness.",
)
async def health_check(request: Request) -> HealthResponse:
    """Return current service health status."""
    model = request.app.state.model
    settings = request.app.state.settings

    return HealthResponse(
        status="healthy" if model.is_loaded else "degraded",
        version=settings.app_version,
        model_loaded=model.is_loaded,
    )
