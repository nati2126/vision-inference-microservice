"""Tests for the detection endpoint."""

import io
from pathlib import Path

import numpy as np
import pytest
from httpx import AsyncClient


def _create_test_image_bytes(width: int = 640, height: int = 480) -> bytes:
    """Generate a synthetic JPEG image for testing.

    Creates a simple gradient image that the model can process (it won't
    detect much, but the pipeline should execute without errors).
    """
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv-python-headless is required for this test")

    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()


@pytest.mark.asyncio
async def test_detect_returns_200_with_valid_image(client: AsyncClient) -> None:
    """POST /api/v1/detect with a valid JPEG should return 200."""
    image_bytes = _create_test_image_bytes()

    response = await client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )

    assert response.status_code == 200

    body = response.json()
    assert "detections" in body
    assert "metadata" in body
    assert isinstance(body["detections"], list)
    assert body["metadata"]["detections_count"] >= 0
    assert body["metadata"]["inference_time_ms"] > 0


@pytest.mark.asyncio
async def test_detect_rejects_non_image_file(client: AsyncClient) -> None:
    """POST /api/v1/detect with a text file should return 400."""
    response = await client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_detect_rejects_corrupt_image(client: AsyncClient) -> None:
    """POST /api/v1/detect with corrupt image bytes should return 400."""
    response = await client.post(
        "/api/v1/detect",
        files={"file": ("bad.jpg", io.BytesIO(b"\x00\x01\x02"), "image/jpeg")},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_detect_response_schema(client: AsyncClient) -> None:
    """Verify detection response matches DetectionResponse schema."""
    image_bytes = _create_test_image_bytes()

    response = await client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )

    body = response.json()

    # Metadata fields
    metadata = body["metadata"]
    assert "image_width" in metadata
    assert "image_height" in metadata
    assert "inference_time_ms" in metadata
    assert "detections_count" in metadata

    # If there are detections, verify their shape
    for det in body["detections"]:
        assert "label" in det
        assert "confidence" in det
        assert "bbox" in det
        assert all(k in det["bbox"] for k in ("x_min", "y_min", "x_max", "y_max"))
        assert 0.0 <= det["confidence"] <= 1.0
