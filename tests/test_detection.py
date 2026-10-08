"""/detect endpoint."""

import io

import numpy as np
import pytest
from httpx import AsyncClient

from app.services.detection_service import _MAX_IMAGE_SIZE_BYTES


def _create_test_image_bytes(width: int = 640, height: int = 480) -> bytes:
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv-python-headless is required for this test")

    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_returns_200_with_valid_image(client: AsyncClient) -> None:
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


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_rejects_non_image_file(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )

    assert response.status_code == 400


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_rejects_corrupt_image(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/detect",
        files={"file": ("bad.jpg", io.BytesIO(b"\x00\x01\x02"), "image/jpeg")},
    )

    assert response.status_code == 400


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_response_schema(client: AsyncClient) -> None:
    image_bytes = _create_test_image_bytes()

    response = await client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )

    body = response.json()

    metadata = body["metadata"]
    assert "image_width" in metadata
    assert "image_height" in metadata
    assert "inference_time_ms" in metadata
    assert "detections_count" in metadata

    for det in body["detections"]:
        assert "label" in det
        assert "confidence" in det
        assert "bbox" in det
        assert all(k in det["bbox"] for k in ("x_min", "y_min", "x_max", "y_max"))
        assert 0.0 <= det["confidence"] <= 1.0


@pytest.mark.asyncio(loop_scope="session")
async def test_error_response_matches_documented_schema(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )

    assert response.status_code == 400

    body = response.json()
    assert "error" in body, "ErrorResponse declares `error`, not FastAPI's `detail`"
    assert isinstance(body["error"], str)
    assert "Unsupported file type" in body["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_rejects_oversized_image(client: AsyncClient) -> None:
    oversized = bytes(_MAX_IMAGE_SIZE_BYTES + 1)

    response = await client.post(
        "/api/v1/detect",
        files={"file": ("big.jpg", io.BytesIO(oversized), "image/jpeg")},
    )

    assert response.status_code == 400
    assert "exceeds maximum size" in response.json()["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_requires_a_file(client: AsyncClient) -> None:
    response = await client.post("/api/v1/detect")

    assert response.status_code == 422
    assert response.json()["error"] == "Request validation failed."


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_rejects_empty_file(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/detect",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_rejects_image_content_type_with_non_image_bytes(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/detect",
        files={"file": ("fake.png", io.BytesIO(b"hello, not a png"), "image/png")},
    )

    assert response.status_code == 400
    assert "Unable to decode" in response.json()["error"]


@pytest.mark.asyncio(loop_scope="session")
async def test_onnx_backend_serves_the_same_response_schema(
    onnx_client: AsyncClient, client: AsyncClient
) -> None:
    from ultralytics.utils import ASSETS

    image_bytes = (ASSETS / "bus.jpg").read_bytes()
    files = {"file": ("bus.jpg", image_bytes, "image/jpeg")}

    onnx_body = (await onnx_client.post("/api/v1/detect", files=files)).json()
    torch_body = (await client.post("/api/v1/detect", files=files)).json()

    assert onnx_body.keys() == torch_body.keys()
    assert onnx_body["metadata"].keys() == torch_body["metadata"].keys()
    assert onnx_body["detections"][0].keys() == torch_body["detections"][0].keys()
    assert onnx_body["metadata"]["image_width"] == 810
    assert onnx_body["metadata"]["image_height"] == 1080
    assert {d["label"] for d in onnx_body["detections"]} >= {"bus", "person"}


@pytest.mark.asyncio(loop_scope="session")
async def test_onnx_backend_rejects_bad_input_identically(onnx_client: AsyncClient) -> None:
    response = await onnx_client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["error"]
