"""YOLOv8 model wrapper.

Encapsulates model loading and raw inference so the rest of the application
never imports ``ultralytics`` directly. This makes it trivial to swap the
model backend later (e.g. ONNX Runtime, TensorRT) without touching business
logic.
"""

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from app.core.logging import get_logger

logger = get_logger(__name__)


class YOLOModel:
    """Thread-safe, singleton-style wrapper around a YOLOv8 model.

    Attributes:
        model: The loaded ``ultralytics.YOLO`` instance.
        model_name: Filename / path of the model weights.
        device: Compute device (``cpu``, ``cuda``, ``mps``).
        confidence_threshold: Minimum confidence for detections.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model: YOLO | None = None

    # ── Lifecycle ────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights into memory.

        Called once during application startup via the lifespan manager.
        """
        logger.info(
            "loading_model",
            model_name=self.model_name,
            device=self.device,
        )
        self._model = YOLO(self.model_name)
        # Warm-up: run a dummy inference to trigger JIT compilation / graph
        # optimisation. This reduces latency for the first real request.
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(
            source=dummy,
            device=self.device,
            verbose=False,
        )
        logger.info("model_loaded", model_name=self.model_name)

    def unload(self) -> None:
        """Release model resources."""
        self._model = None
        logger.info("model_unloaded", model_name=self.model_name)

    # ── Inference ────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run object detection on a single image.

        Args:
            image: BGR or RGB numpy array (H×W×3).

        Returns:
            List of detection dicts, each containing:
            ``label``, ``confidence``, ``bbox`` (xyxy format).

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    {
                        "label": result.names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox": {
                            "x_min": round(xyxy[0], 2),
                            "y_min": round(xyxy[1], 2),
                            "x_max": round(xyxy[2], 2),
                            "y_max": round(xyxy[3], 2),
                        },
                    }
                )

        return detections

    @property
    def is_loaded(self) -> bool:
        """Check whether the model weights are currently in memory."""
        return self._model is not None
