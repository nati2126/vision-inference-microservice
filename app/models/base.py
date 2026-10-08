"""The contract every inference backend implements.

The service layer only ever talks to :class:`InferenceBackend`, so swapping
PyTorch for ONNX Runtime, OpenVINO or TensorRT is a configuration change, not
a code change. Backends return plain :class:`Detection` records in the pixel
coordinates of the *original* image; turning them into the HTTP response is
the service's job.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object, in original-image pixel coordinates (xyxy)."""

    class_id: int
    label: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the shape of ``app.schemas.detection.Detection``."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x_min": round(self.x_min, 2),
                "y_min": round(self.y_min, 2),
                "x_max": round(self.x_max, 2),
                "y_max": round(self.y_max, 2),
            },
        }


class InferenceBackend(ABC):
    """A loaded detector that maps a BGR image to a list of detections.

    Subclasses set ``name`` and implement ``load``, ``predict`` and
    ``precision``. Thresholds are fixed at construction so that every
    backend can be driven with identical settings (the benchmark relies on
    this for a fair accuracy comparison).

    Args:
        model_path: Path to the model artifact (``.pt``, ``.onnx``, ``.xml``,
            ``.engine``).
        device: ``"cpu"`` or ``"cuda"`` (``"mps"`` for PyTorch only).
        confidence_threshold: Minimum class score for a detection.
        iou_threshold: IoU above which NMS suppresses the weaker box.
        max_detections: Upper bound on detections returned per image.
    """

    name: str = "base"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    @abstractmethod
    def load(self) -> None:
        """Load the artifact and warm the backend up."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model and any device memory it holds."""

    @abstractmethod
    def predict(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        """Detect objects in a BGR ``(H, W, 3)`` uint8 image.

        Raises:
            RuntimeError: If called before :meth:`load`.
        """

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether :meth:`load` has completed."""

    @property
    @abstractmethod
    def precision(self) -> str:
        """Numeric precision of the loaded artifact: ``fp32``, ``fp16`` or ``int8``.

        Derived from the artifact itself, never from configuration, so the
        value reported by ``/health`` cannot drift from what actually runs.
        """

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call .load() first.")
