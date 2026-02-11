"""Detection service — orchestrates image decoding and model inference.

This layer owns the *business logic*: it knows how to decode an uploaded
file into an image array, delegates inference to the model layer, and
packages the response. Route handlers should be thin wrappers around
service methods.
"""

import time
from typing import Any

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.logging import get_logger
from app.models.yolo_model import YOLOModel

logger = get_logger(__name__)

# Maximum accepted image size in bytes (10 MB)
_MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


class DetectionService:
    """Stateless service that bridges the API and model layers.

    Args:
        model: A loaded ``YOLOModel`` instance (injected via the lifespan
            context — never constructed here).
    """

    def __init__(self, model: YOLOModel) -> None:
        self._model = model

    async def detect(self, file: UploadFile) -> dict[str, Any]:
        """Run detection on an uploaded image file.

        Args:
            file: The uploaded image (JPEG / PNG).

        Returns:
            A dict containing ``detections`` and ``metadata``.

        Raises:
            ValueError: If the file is too large or cannot be decoded as an image.
        """
        contents = await file.read()

        # ── Guard: file size ─────────────────────────────────
        if len(contents) > _MAX_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"Image exceeds maximum size of "
                f"{_MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB."
            )

        # ── Decode image ─────────────────────────────────────
        image = self._decode_image(contents)

        # ── Inference ────────────────────────────────────────
        start = time.perf_counter()
        detections = self._model.predict(image)
        inference_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "detection_complete",
            detections_count=len(detections),
            inference_ms=inference_ms,
            image_shape=image.shape,
        )

        return {
            "detections": detections,
            "metadata": {
                "image_width": image.shape[1],
                "image_height": image.shape[0],
                "inference_time_ms": inference_ms,
                "detections_count": len(detections),
            },
        }

    # ── Private helpers ──────────────────────────────────────

    @staticmethod
    def _decode_image(raw_bytes: bytes) -> np.ndarray:
        """Decode raw file bytes into a BGR numpy array.

        Raises:
            ValueError: If OpenCV cannot decode the payload.
        """
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "Unable to decode the uploaded file as an image. "
                "Supported formats: JPEG, PNG, BMP, TIFF."
            )
        return image
