"""Decodes uploads and runs inference."""

import asyncio
import contextlib
import io
import time
from dataclasses import dataclass, field
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.core.logging import get_logger
from app.models.base import Detection, InferenceBackend

logger = get_logger(__name__)

_READ_CHUNK_BYTES = 64 * 1024
_DECODE_ERROR = "Unable to decode the uploaded file as an image. Supported formats: JPEG, PNG, BMP, TIFF, WebP."


@dataclass
class _Pending:
    inputs: Any
    future: asyncio.Future[tuple[Any, float, int]] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


class DetectionService:
    """Runs model calls on one worker task that batches requests already queued.

    With batch_timeout_ms=0 nothing waits for a batch to fill: a request that
    arrives while the model is idle runs alone, and requests that queue up
    behind a running call share the next one, up to max_batch_size. With a
    batch size of 1 a lock serialises the model call instead, which measured
    a few percent faster than the worker.
    """

    def __init__(
        self,
        model: InferenceBackend,
        max_upload_bytes: int = 10 * 1024 * 1024,
        max_image_pixels: int = 50_000_000,
        max_batch_size: int = 8,
        batch_timeout_ms: float = 0.0,
    ) -> None:
        self._model = model
        self._max_upload_bytes = max_upload_bytes
        self._max_image_pixels = max_image_pixels
        model_limit = model.max_batch_size
        self._max_batch = max(1, min(max_batch_size, model_limit or max_batch_size))
        self._batch_timeout_s = batch_timeout_ms / 1000
        self._queue: asyncio.Queue[_Pending] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    @property
    def max_batch_size(self) -> int:
        return self._max_batch

    def warm_up(self) -> None:
        """Run each batch size once, so no request pays for per-shape setup.

        OpenVINO prepares each new input shape on first use (about 80-100 ms
        each here).
        """
        if self._max_batch == 1:
            return
        start = time.perf_counter()
        inputs = self._model.preprocess(np.zeros((64, 64, 3), dtype=np.uint8))
        for size in range(2, self._max_batch + 1):
            self._model.infer_batch([inputs] * size)
        logger.info(
            "batch_sizes_warmed_up",
            max_batch_size=self._max_batch,
            seconds=round(time.perf_counter() - start, 2),
        )

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = self._queue = None

    async def detect(self, file: UploadFile) -> dict[str, Any]:
        """Raises ValueError for oversized or undecodable uploads."""
        contents = await self._read_capped(file)
        image, inputs, preprocess_s = await run_in_threadpool(self._prepare, contents)

        outputs, infer_s, batch_size = await self._submit(inputs)

        start = time.perf_counter()
        detections: list[Detection] = await run_in_threadpool(
            self._model.postprocess, outputs, inputs
        )
        postprocess_s = time.perf_counter() - start

        # Excludes queueing; the model call is timed for the whole batch.
        inference_ms = round((preprocess_s + infer_s + postprocess_s) * 1000, 2)
        logger.info(
            "detection_complete",
            detections_count=len(detections),
            inference_ms=inference_ms,
            batch_size=batch_size,
            image_shape=image.shape,
        )
        return {
            "detections": [detection.to_dict() for detection in detections],
            "metadata": {
                "image_width": image.shape[1],
                "image_height": image.shape[0],
                "inference_time_ms": inference_ms,
                "detections_count": len(detections),
            },
        }

    async def _submit(self, inputs: Any) -> tuple[Any, float, int]:
        if self._max_batch == 1:
            async with self._lock:
                start = time.perf_counter()
                output = await run_in_threadpool(self._model.infer, inputs)
                return output, time.perf_counter() - start, 1
        if self._worker is None:
            # Created on first use so the queue and task belong to the serving loop.
            self._queue = asyncio.Queue()
            self._worker = asyncio.create_task(self._run_batches())
        assert self._queue is not None
        pending = _Pending(inputs)
        self._queue.put_nowait(pending)
        return await pending.future

    async def _run_batches(self) -> None:
        assert self._queue is not None
        while True:
            batch = [await self._queue.get()]
            await self._fill(batch)
            batch = [p for p in batch if not p.future.done()]
            if not batch:
                continue

            start = time.perf_counter()
            try:
                outputs = await run_in_threadpool(
                    self._model.infer_batch, [p.inputs for p in batch]
                )
                infer_s = time.perf_counter() - start
                results = list(zip(batch, outputs, strict=True))
            except Exception as exc:
                # The worker must survive, or every later request would hang.
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(exc)
                continue

            for p, output in results:
                if not p.future.done():
                    p.future.set_result((output, infer_s, len(batch)))

    async def _fill(self, batch: list[_Pending]) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._batch_timeout_s
        while len(batch) < self._max_batch:
            remaining = deadline - loop.time()
            try:
                if remaining <= 0:
                    batch.append(self._queue.get_nowait())
                else:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                return

    def _prepare(self, contents: bytes) -> tuple[npt.NDArray[np.uint8], Any, float]:
        image = self._decode_image(contents)
        start = time.perf_counter()
        inputs = self._model.preprocess(image)
        return image, inputs, time.perf_counter() - start

    async def _read_capped(self, file: UploadFile) -> bytes:
        limit = self._max_upload_bytes
        message = f"Image exceeds maximum size of {limit} bytes."
        if file.size is not None and file.size > limit:
            raise ValueError(message)

        chunks: list[bytes] = []
        total = 0
        while chunk := await file.read(_READ_CHUNK_BYTES):
            total += len(chunk)
            if total > limit:
                raise ValueError(message)
            chunks.append(chunk)
        return b"".join(chunks)

    def _decode_image(self, raw_bytes: bytes) -> npt.NDArray[np.uint8]:
        if not raw_bytes:
            raise ValueError("Uploaded file is empty.")

        # A few-KB PNG can declare dimensions that take gigabytes to decode.
        limit = self._max_image_pixels
        try:
            with Image.open(io.BytesIO(raw_bytes)) as header:
                width, height = header.size
        except Image.DecompressionBombError as exc:
            raise ValueError(f"Image exceeds the maximum of {limit} pixels.") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(_DECODE_ERROR) from exc
        if width * height > limit:
            raise ValueError(
                f"Image is {width}x{height}, which exceeds the maximum of {limit} pixels."
            )

        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cast("npt.NDArray[np.uint8] | None", cv2.imdecode(np_arr, cv2.IMREAD_COLOR))
        if image is None:
            raise ValueError(_DECODE_ERROR)
        return image
