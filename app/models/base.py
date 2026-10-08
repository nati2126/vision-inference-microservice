"""Interface shared by all inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection in original-image pixels (xyxy)."""

    class_id: int
    label: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_dict(self) -> dict[str, Any]:
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
    """A detector that maps a BGR image to a list of detections."""

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
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def predict(self, image: npt.NDArray[np.uint8]) -> list[Detection]: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def precision(self) -> str:
        """fp32, fp16 or int8, read from the artifact rather than the config."""

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call .load() first.")
