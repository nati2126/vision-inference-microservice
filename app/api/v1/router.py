"""V1 router — aggregates all v1 endpoint routers.

Adding a new endpoint group is a one-liner:
    ``router.include_router(new_router, prefix="/new", tags=["new"])``
"""

from fastapi import APIRouter

from app.api.v1.endpoints import detection, health

router = APIRouter(prefix="/api/v1")

router.include_router(health.router, tags=["Health"])
router.include_router(detection.router, tags=["Detection"])
