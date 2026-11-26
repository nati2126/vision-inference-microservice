"""DetectionService with a stub model."""

import asyncio
import io
import struct
import threading
import time
import zlib
from collections.abc import AsyncIterator, Callable
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest
import pytest_asyncio
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


ServiceFactory = Callable[..., DetectionService]


@pytest_asyncio.fixture(loop_scope="session")
async def make_service() -> AsyncIterator[ServiceFactory]:
    created: list[DetectionService] = []

    def make(**kwargs: Any) -> DetectionService:
        created.append(DetectionService(**kwargs))
        return created[-1]

    yield make
    for service in created:
        await service.close()


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
async def test_detect_reports_image_dimensions(make_service: ServiceFactory) -> None:
    service = make_service(model=_StubModel())

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
async def test_oversized_upload_is_rejected(make_service: ServiceFactory) -> None:
    model = _StubModel()
    service = make_service(model=model, max_upload_bytes=1024)

    with pytest.raises(ValueError, match="exceeds maximum size"):
        await service.detect(_upload(b"\x00" * 1025))

    assert model.calls == [], "oversized upload must not reach the model"


@pytest.mark.asyncio(loop_scope="session")
async def test_upload_at_the_limit_is_not_rejected_for_size(make_service: ServiceFactory) -> None:
    service = make_service(model=_StubModel(), max_upload_bytes=1024)

    with pytest.raises(ValueError, match="Unable to decode"):
        await service.detect(_upload(b"\x00" * 1024))


@pytest.mark.asyncio(loop_scope="session")
async def test_image_over_the_pixel_limit_is_rejected_before_decoding(
    make_service: ServiceFactory,
) -> None:
    model = _StubModel()
    service = make_service(model=model, max_image_pixels=64 * 64)

    with pytest.raises(ValueError, match="exceeds the maximum of 4096 pixels"):
        await service.detect(_upload(_jpeg_bytes(65, 64)))

    assert model.calls == []


@pytest.mark.asyncio(loop_scope="session")
async def test_tiny_png_declaring_huge_dimensions_is_rejected(
    make_service: ServiceFactory,
) -> None:
    model = _StubModel()
    service = make_service(model=model)

    with pytest.raises(ValueError, match="exceeds the maximum of 50000000 pixels"):
        await service.detect(_upload(_tiny_png_claiming(100_000, 100_000), "bomb.png"))

    assert model.calls == []


@pytest.mark.asyncio(loop_scope="session")
async def test_image_at_the_pixel_limit_is_accepted(make_service: ServiceFactory) -> None:
    service = make_service(model=_StubModel(), max_image_pixels=64 * 64)

    result = await service.detect(_upload(_jpeg_bytes(64, 64)))

    assert result["metadata"]["detections_count"] == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_undecodable_upload_raises_value_error(make_service: ServiceFactory) -> None:
    service = make_service(model=_StubModel())

    with pytest.raises(ValueError, match="Unable to decode"):
        await service.detect(_upload(b"\x00\x01\x02"))


@pytest.mark.asyncio(loop_scope="session")
async def test_empty_upload_raises_value_error(make_service: ServiceFactory) -> None:
    service = make_service(model=_StubModel())

    with pytest.raises(ValueError, match="empty"):
        await service.detect(_upload(b""))


@pytest.mark.asyncio(loop_scope="session")
async def test_preprocessing_overlaps_while_inference_is_serialised(
    make_service: ServiceFactory,
) -> None:
    # Each preprocess waits for the other, so this deadlocks if they run one at a time.
    model = _StubModel(preprocess_barrier=threading.Barrier(2, timeout=10))
    service = make_service(model=model)

    await asyncio.gather(
        service.detect(_upload(_jpeg_bytes())), service.detect(_upload(_jpeg_bytes()))
    )

    assert len(model.calls) == 2
    assert model.max_concurrent_infer == 1


class _BatchingStub(_StubModel):
    """Echoes each input's size as a box; the first call can be held until released."""

    def __init__(self, limit: int | None = None, hold_first_call: bool = False) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.fail_next = False
        self._limit = limit
        self.started = threading.Event()
        self.release = threading.Event()
        if not hold_first_call:
            self.release.set()

    @property
    def max_batch_size(self) -> int | None:
        return self._limit

    def infer_batch(self, batch: list[npt.NDArray[np.uint8]]) -> list[list[Detection]]:
        self.started.set()
        assert self.release.wait(10)
        self.batch_sizes.append(len(batch))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("model call failed")
        return [
            [Detection(0, "person", 0.9, 0.0, 0.0, float(i.shape[1]), float(i.shape[0]))]
            for i in batch
        ]


async def _wait_until_queued(service: DetectionService, count: int) -> None:
    for _ in range(1000):
        if service._queue is not None and service._queue.qsize() >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{count} requests never queued")


async def _detect_widths(
    service: DetectionService, model: _BatchingStub, first: int, rest: list[int]
) -> list[dict[str, object]]:
    """Run one request, queue the rest behind its held model call, then release."""
    head = asyncio.create_task(service.detect(_upload(_jpeg_bytes(first, 32))))
    assert await asyncio.to_thread(model.started.wait, 10)
    tail = [asyncio.create_task(service.detect(_upload(_jpeg_bytes(w, 32)))) for w in rest]
    await _wait_until_queued(service, len(rest))
    model.release.set()
    return list(await asyncio.gather(head, *tail))


@pytest.mark.asyncio(loop_scope="session")
async def test_requests_queued_behind_a_model_call_share_the_next_one(
    make_service: ServiceFactory,
) -> None:
    model = _BatchingStub(hold_first_call=True)
    service = make_service(model=model, max_batch_size=8)

    results = await _detect_widths(service, model, 40, [41, 42, 43])

    assert model.batch_sizes == [1, 3]
    for result in results:
        box = result["detections"][0]["bbox"]  # type: ignore[index]
        assert box["x_max"] == result["metadata"]["image_width"]  # type: ignore[index]


@pytest.mark.asyncio(loop_scope="session")
async def test_batches_never_exceed_max_batch_size(make_service: ServiceFactory) -> None:
    model = _BatchingStub(hold_first_call=True)
    service = make_service(model=model, max_batch_size=2)

    await _detect_widths(service, model, 40, [41, 42, 43, 44, 45])

    assert model.batch_sizes == [1, 2, 2, 1]


@pytest.mark.asyncio(loop_scope="session")
async def test_the_model_limit_caps_the_configured_batch_size(
    make_service: ServiceFactory,
) -> None:
    assert make_service(model=_BatchingStub(limit=1), max_batch_size=8).max_batch_size == 1
    assert make_service(model=_BatchingStub(limit=4), max_batch_size=8).max_batch_size == 4
    assert make_service(model=_BatchingStub(limit=None), max_batch_size=8).max_batch_size == 8


@pytest.mark.asyncio(loop_scope="session")
async def test_a_failed_model_call_fails_its_batch_and_the_service_recovers(
    make_service: ServiceFactory,
) -> None:
    model = _BatchingStub(hold_first_call=True)
    model.fail_next = True
    service = make_service(model=model, max_batch_size=8)
    head = asyncio.create_task(service.detect(_upload(_jpeg_bytes(40, 32))))
    assert await asyncio.to_thread(model.started.wait, 10)
    model.release.set()

    with pytest.raises(RuntimeError, match="model call failed"):
        await head
    result = await service.detect(_upload(_jpeg_bytes(41, 32)))

    assert result["metadata"]["image_width"] == 41


@pytest.mark.asyncio(loop_scope="session")
async def test_warm_up_runs_every_batch_size_once(make_service: ServiceFactory) -> None:
    model = _BatchingStub()
    service = make_service(model=model, max_batch_size=4)

    service.warm_up()

    assert model.batch_sizes == [2, 3, 4]


@pytest.mark.asyncio(loop_scope="session")
async def test_warm_up_is_a_no_op_without_batching(make_service: ServiceFactory) -> None:
    model = _BatchingStub(limit=1)
    service = make_service(model=model, max_batch_size=8)

    service.warm_up()

    assert model.batch_sizes == []
