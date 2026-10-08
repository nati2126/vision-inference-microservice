"""Decodes uploads and runs inference."""

import asyncio
import time
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.logging import get_logger
from app.models.base import InferenceBackend

logger = get_logger(__name__)

_MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

_READ_CHUNK_BYTES = 64 * 1024


class DetectionService:
    def __init__(self, model: InferenceBackend) -> None:
        self._model = model
        # Backends are not thread-safe (shared infer request / I/O buffers).
        self._inference_lock = asyncio.Lock()

    async def detect(self, file: UploadFile) -> dict[str, Any]:
        """Raises ValueError for oversized or undecodable uploads."""
        contents = await self._read_capped(file)

        image = self._decode_image(contents)

        # Off the event loop; timed inside the lock so queueing isn't counted.
        async with self._inference_lock:
            start = time.perf_counter()
            detections = await run_in_threadpool(self._model.predict, image)
            inference_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "detection_complete",
            detections_count=len(detections),
            inference_ms=inference_ms,
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

    @staticmethod
    async def _read_capped(file: UploadFile) -> bytes:
        limit_mb = _MAX_IMAGE_SIZE_BYTES // (1024 * 1024)

        if file.size is not None and file.size > _MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Image exceeds maximum size of {limit_mb} MB.")

        chunks: list[bytes] = []
        total = 0
        while chunk := await file.read(_READ_CHUNK_BYTES):
            total += len(chunk)
            if total > _MAX_IMAGE_SIZE_BYTES:
                raise ValueError(f"Image exceeds maximum size of {limit_mb} MB.")
            chunks.append(chunk)

        return b"".join(chunks)

    @staticmethod
    def _decode_image(raw_bytes: bytes) -> npt.NDArray[np.uint8]:
        if not raw_bytes:
            raise ValueError("Uploaded file is empty.")

        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cast("npt.NDArray[np.uint8] | None", cv2.imdecode(np_arr, cv2.IMREAD_COLOR))
        if image is None:
            raise ValueError(
                "Unable to decode the uploaded file as an image. "
                "Supported formats: JPEG, PNG, BMP, TIFF."
            )
        return image
