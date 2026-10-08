"""Detection service — orchestrates image decoding and model inference.

This layer owns the *business logic*: it knows how to decode an uploaded
file into an image array, delegates inference to the model layer, and
packages the response. Route handlers should be thin wrappers around
service methods.
"""

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

# Maximum accepted image size in bytes (10 MB)
_MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

# Size of each chunk when streaming the upload off the wire.
_READ_CHUNK_BYTES = 64 * 1024


class DetectionService:
    """Stateless service that bridges the API and model layers.

    Args:
        model: A loaded ``InferenceBackend`` (injected via the lifespan
            context — never constructed here).
    """

    def __init__(self, model: InferenceBackend) -> None:
        self._model = model
        # Inference runs off the event loop (see ``detect``), but backends
        # are not all safe to call from several threads at once (an
        # ultralytics model, a single OpenVINO infer request, TensorRT's
        # pre-bound I/O buffers), so calls into them are serialised. The lock costs nothing
        # while a single request is in flight and prevents interleaved access
        # under concurrency.
        self._inference_lock = asyncio.Lock()

    async def detect(self, file: UploadFile) -> dict[str, Any]:
        """Run detection on an uploaded image file.

        Args:
            file: The uploaded image (JPEG / PNG).

        Returns:
            A dict containing ``detections`` and ``metadata``.

        Raises:
            ValueError: If the file is too large or cannot be decoded as an image.
        """
        contents = await self._read_capped(file)

        # ── Decode image ─────────────────────────────────────
        image = self._decode_image(contents)

        # ── Inference ────────────────────────────────────────
        # ``InferenceBackend.predict`` is synchronous and CPU/GPU-bound. Awaiting it
        # directly would pin the event loop for the whole inference, stalling
        # every other request on this worker — including health probes. Run it
        # on a worker thread instead so the loop stays responsive.
        # The timer starts inside the lock: ``inference_time_ms`` is documented
        # as model time, so it must not absorb the wait for a busy model.
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

    # ── Private helpers ──────────────────────────────────────

    @staticmethod
    async def _read_capped(file: UploadFile) -> bytes:
        """Read the upload, refusing anything over the size cap.

        The cap is enforced *before* the bytes are accumulated. Reading the
        whole upload first and checking its length afterwards means an
        oversized body is fully materialised before it can be rejected, so
        the guard does not actually protect anything.

        Raises:
            ValueError: If the upload exceeds ``_MAX_IMAGE_SIZE_BYTES``.
        """
        limit_mb = _MAX_IMAGE_SIZE_BYTES // (1024 * 1024)

        # Starlette populates ``size`` from the multipart parser, so an
        # oversized upload is usually refused without reading a single chunk.
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
        """Decode raw file bytes into a BGR numpy array.

        Raises:
            ValueError: If OpenCV cannot decode the payload.
        """
        if not raw_bytes:
            # cv2.imdecode asserts on an empty buffer and raises cv2.error,
            # which would escape as a 500 rather than a 400.
            raise ValueError("Uploaded file is empty.")

        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        # ``IMREAD_COLOR`` always yields an 8-bit 3-channel BGR array; the cv2
        # stubs declare a wider dtype than that flag can actually produce.
        image = cast("npt.NDArray[np.uint8] | None", cv2.imdecode(np_arr, cv2.IMREAD_COLOR))
        if image is None:
            raise ValueError(
                "Unable to decode the uploaded file as an image. "
                "Supported formats: JPEG, PNG, BMP, TIFF."
            )
        return image
