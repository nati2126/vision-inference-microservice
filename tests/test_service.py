"""DetectionService with a stub model."""

import asyncio
import io
import struct
import threading
import time
import zlib

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from fastapi import UploadFile

from app.models.base import Detection, InferenceBackend
from app.services.detection_service import DetectionService


class _StubModel(InferenceBackend):
    name = "stub"

    def __init__(self, preprocess_barrier: threading.Barrier | None = None) -> None:
        super().__init__("stub")
        self.calls: list[tuple[int, int]] = []
        self.max_concurrent_infer = 0
        self._active = 0
        self._counter_lock = threading.Lock()
        self._barrier = preprocess_barrier

    def load(self) -> None: ...

    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def precision(self) -> str:
        return "fp32"

    def preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self._barrier is not None:
            self._barrier.wait()
        return image

    def infer(self, inputs: npt.NDArray[np.uint8]) -> list[Detection]:
        with self._counter_lock:
            self._active += 1
            self.max_concurrent_infer = max(self.max_concurrent_infer, self._active)
        time.sleep(0.02)
        with self._counter_lock:
            self._active -= 1
        self.calls.append((inputs.shape[1], inputs.shape[0]))
        return [Detection(0, "person", 0.9, 1.0, 2.0, 3.0, 4.0)]


def _upload(data: bytes, filename: str = "test.jpg") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(data))


def _jpeg_bytes(width: int = 64, height: int = 32) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", image)
    return bytes(buffer.tobytes())


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def _tiny_png_claiming(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(b"\x00" * 64))
        + _png_chunk(b"IEND", b"")
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_reports_image_dimensions() -> None:
    service = DetectionService(model=_StubModel())

    result = await service.detect(_upload(_jpeg_bytes(64, 32)))

    assert result["metadata"]["image_width"] == 64
    assert result["metadata"]["image_height"] == 32
    assert result["metadata"]["detections_count"] == 1
    assert result["detections"][0] == {
        "label": "person",
        "confidence": 0.9,
        "bbox": {"x_min": 1.0, "y_min": 2.0, "x_max": 3.0, "y_max": 4.0},
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_oversized_upload_is_rejected() -> None:
    model = _StubModel()
    service = DetectionService(model=model, max_upload_bytes=1024)

    with pytest.raises(ValueError, match="exceeds maximum size"):
        await service.detect(_upload(b"\x00" * 1025))

    assert model.calls == [], "oversized upload must not reach the model"


@pytest.mark.asyncio(loop_scope="session")
async def test_upload_at_the_limit_is_not_rejected_for_size() -> None:
    service = DetectionService(model=_StubModel(), max_upload_bytes=1024)

    with pytest.raises(ValueError, match="Unable to decode"):
        await service.detect(_upload(b"\x00" * 1024))


@pytest.mark.asyncio(loop_scope="session")
async def test_image_over_the_pixel_limit_is_rejected_before_decoding() -> None:
    model = _StubModel()
    service = DetectionService(model=model, max_image_pixels=64 * 64)

    with pytest.raises(ValueError, match="exceeds the maximum of 4096 pixels"):
        await service.detect(_upload(_jpeg_bytes(65, 64)))

    assert model.calls == []


@pytest.mark.asyncio(loop_scope="session")
async def test_tiny_png_declaring_huge_dimensions_is_rejected() -> None:
    model = _StubModel()
    service = DetectionService(model=model)

    with pytest.raises(ValueError, match="exceeds the maximum of 50000000 pixels"):
        await service.detect(_upload(_tiny_png_claiming(100_000, 100_000), "bomb.png"))

    assert model.calls == []


@pytest.mark.asyncio(loop_scope="session")
async def test_image_at_the_pixel_limit_is_accepted() -> None:
    service = DetectionService(model=_StubModel(), max_image_pixels=64 * 64)

    result = await service.detect(_upload(_jpeg_bytes(64, 64)))

    assert result["metadata"]["detections_count"] == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_undecodable_upload_raises_value_error() -> None:
    service = DetectionService(model=_StubModel())

    with pytest.raises(ValueError, match="Unable to decode"):
        await service.detect(_upload(b"\x00\x01\x02"))


@pytest.mark.asyncio(loop_scope="session")
async def test_empty_upload_raises_value_error() -> None:
    service = DetectionService(model=_StubModel())

    with pytest.raises(ValueError, match="empty"):
        await service.detect(_upload(b""))


@pytest.mark.asyncio(loop_scope="session")
async def test_preprocessing_overlaps_while_inference_is_serialised() -> None:
    # Each preprocess waits for the other, so this deadlocks if they run one at a time.
    model = _StubModel(preprocess_barrier=threading.Barrier(2, timeout=10))
    service = DetectionService(model=model)

    await asyncio.gather(
        service.detect(_upload(_jpeg_bytes())), service.detect(_upload(_jpeg_bytes()))
    )

    assert len(model.calls) == 2
    assert model.max_concurrent_infer == 1
