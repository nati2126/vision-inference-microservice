"""Unit tests for DetectionService, using a stand-in model.

These exercise the service layer without loading real weights, so they stay
fast and cover the error paths the endpoint tests cannot reach directly.
"""

import io
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from fastapi import UploadFile

from app.services.detection_service import _MAX_IMAGE_SIZE_BYTES, DetectionService


class _StubModel:
    """Minimal stand-in for YOLOModel that records what it was given."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def predict(self, image: npt.NDArray[np.uint8]) -> list[dict[str, Any]]:
        self.calls.append((image.shape[1], image.shape[0]))
        return [
            {
                "label": "person",
                "confidence": 0.9,
                "bbox": {"x_min": 1.0, "y_min": 2.0, "x_max": 3.0, "y_max": 4.0},
            }
        ]


def _upload(data: bytes, filename: str = "test.jpg") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(data))


def _jpeg_bytes(width: int = 64, height: int = 32) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", image)
    return bytes(buffer.tobytes())


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_reports_image_dimensions() -> None:
    """Metadata reflects the decoded image, not the request."""
    service = DetectionService(model=_StubModel())  # type: ignore[arg-type]

    result = await service.detect(_upload(_jpeg_bytes(64, 32)))

    assert result["metadata"]["image_width"] == 64
    assert result["metadata"]["image_height"] == 32
    assert result["metadata"]["detections_count"] == 1
    assert result["detections"][0]["label"] == "person"


@pytest.mark.asyncio(loop_scope="session")
async def test_oversized_upload_is_rejected() -> None:
    """The size cap is enforced, and the model is never invoked."""
    model = _StubModel()
    service = DetectionService(model=model)  # type: ignore[arg-type]
    oversized = b"\x00" * (_MAX_IMAGE_SIZE_BYTES + 1)

    with pytest.raises(ValueError, match="exceeds maximum size"):
        await service.detect(_upload(oversized))

    assert model.calls == [], "oversized upload must not reach the model"


@pytest.mark.asyncio(loop_scope="session")
async def test_upload_at_the_limit_is_not_rejected_for_size() -> None:
    """A payload exactly at the cap fails on decoding, not on size."""
    service = DetectionService(model=_StubModel())  # type: ignore[arg-type]
    at_limit = b"\x00" * _MAX_IMAGE_SIZE_BYTES

    with pytest.raises(ValueError, match="Unable to decode"):
        await service.detect(_upload(at_limit))


@pytest.mark.asyncio(loop_scope="session")
async def test_undecodable_upload_raises_value_error() -> None:
    service = DetectionService(model=_StubModel())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unable to decode"):
        await service.detect(_upload(b"\x00\x01\x02"))


@pytest.mark.asyncio(loop_scope="session")
async def test_empty_upload_raises_value_error() -> None:
    """An empty body is a ValueError, not the cv2.error imdecode asserts with."""
    service = DetectionService(model=_StubModel())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="empty"):
        await service.detect(_upload(b""))
