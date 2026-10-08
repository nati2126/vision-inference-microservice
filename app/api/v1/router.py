"""API v1 router."""

from fastapi import APIRouter

from app.api.v1.endpoints import detection, health

router = APIRouter(prefix="/api/v1")

router.include_router(health.router, tags=["Health"])
router.include_router(detection.router, tags=["Detection"])
