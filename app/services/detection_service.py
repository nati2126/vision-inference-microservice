"""Decodes uploads and runs inference."""

import asyncio
import io
import time
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


class DetectionService:
    def __init__(
        self,
        model: InferenceBackend,
        max_upload_bytes: int = 10 * 1024 * 1024,
        max_image_pixels: int = 50_000_000,
    ) -> None:
        self._model = model
        self._max_upload_bytes = max_upload_bytes
        self._max_image_pixels = max_image_pixels
        # Only the infer stage holds it; decoding and pre/post-processing overlap.
        self._inference_lock = asyncio.Lock()

    async def detect(self, file: UploadFile) -> dict[str, Any]:
        """Raises ValueError for oversized or undecodable uploads."""
        contents = await self._read_capped(file)
        image, inputs, preprocess_s = await run_in_threadpool(self._prepare, contents)

        async with self._inference_lock:
            start = time.perf_counter()
            outputs = await run_in_threadpool(self._model.infer, inputs)
            infer_s = time.perf_counter() - start

        start = time.perf_counter()
        detections: list[Detection] = await run_in_threadpool(
            self._model.postprocess, outputs, inputs
        )
        postprocess_s = time.perf_counter() - start

        # Excludes time spent waiting for the lock.
        inference_ms = round((preprocess_s + infer_s + postprocess_s) * 1000, 2)
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
