"""POST /detect."""

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from app.core.logging import get_logger
from app.schemas.detection import DetectionResponse, ErrorResponse

logger = get_logger(__name__)

router = APIRouter()

_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}


@router.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Run object detection",
    description="Upload an image file (JPEG, PNG) and receive detected objects.",
)
async def detect_objects(
    request: Request,
    file: UploadFile,
) -> DetectionResponse:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. "
            f"Accepted: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}",
        )

    try:
        service = request.app.state.detection_service
        result = await service.detect(file)
    except ValueError as exc:
        logger.warning("detection_bad_request", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("detection_runtime_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed. Please try again later.",
        ) from exc
    except Exception as exc:
        logger.exception("detection_unexpected_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        ) from exc

    return DetectionResponse(**result)
