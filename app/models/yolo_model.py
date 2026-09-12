"""YOLOv8 model wrapper.

Encapsulates model loading and raw inference so the rest of the application
never imports ``ultralytics`` directly. This makes it trivial to swap the
model backend later (e.g. ONNX Runtime, TensorRT) without touching business
logic.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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
        weights_dir: Directory that bare weight filenames resolve against.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        weights_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.weights_dir = weights_dir
        self._model: YOLO | None = None

    # ── Lifecycle ────────────────────────────────────────────

    def _resolve_weights(self) -> str:
        """Return the path to hand to ultralytics.

        A bare filename (``yolov8n.pt``) is resolved against ``weights_dir``.
        Ultralytics downloads a *relative* filename into the current working
        directory; in the container that is the root-owned ``/app``, so an
        unprivileged process gets ``PermissionError`` on first start. An
        absolute path avoids that and lets the weights persist on a volume.

        An explicit path in ``model_name`` (absolute, or containing a
        separator) is passed through untouched.
        """
        candidate = Path(self.model_name)
        if self.weights_dir is None or candidate.is_absolute() or candidate.parent != Path("."):
            return self.model_name

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        return str(self.weights_dir / candidate.name)

    def load(self) -> None:
        """Load model weights into memory.

        Called once during application startup via the lifespan manager.
        """
        weights = self._resolve_weights()
        logger.info(
            "loading_model",
            model_name=self.model_name,
            weights_path=weights,
            device=self.device,
        )
        self._model = YOLO(weights)
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

    def predict(self, image: npt.NDArray[np.uint8]) -> list[dict[str, Any]]:
        """Run object detection on a single image.

        Args:
            image: BGR or RGB numpy array of shape (H, W, 3).

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
            # Read the batched tensors once rather than iterating ``Boxes``:
            # fewer host/device round-trips, and ``Boxes`` is not typed as
            # iterable so strict mypy rejects the per-box loop.
            coords: list[list[float]] = boxes.xyxy.tolist()
            confidences: list[float] = boxes.conf.tolist()
            class_ids: list[float] = boxes.cls.tolist()

            for xyxy, confidence, class_id in zip(coords, confidences, class_ids, strict=True):
                detections.append(
                    {
                        "label": str(result.names[int(class_id)]),
                        "confidence": round(float(confidence), 4),
                        "bbox": {
                            "x_min": round(float(xyxy[0]), 2),
                            "y_min": round(float(xyxy[1]), 2),
                            "x_max": round(float(xyxy[2]), 2),
                            "y_max": round(float(xyxy[3]), 2),
                        },
                    }
                )

        return detections

    @property
    def is_loaded(self) -> bool:
        """Check whether the model weights are currently in memory."""
        return self._model is not None
