"""Interface shared by all inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

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
    """A detector that maps a BGR image to a list of detections.

    predict() runs three stages. preprocess and postprocess are thread-safe;
    infer and infer_batch are not, so concurrent callers must serialise them.
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
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    def preprocess(self, image: npt.NDArray[np.uint8]) -> Any:
        return image

    @abstractmethod
    def infer(self, inputs: Any) -> Any: ...

    def infer_batch(self, batch: list[Any]) -> list[Any]:
        """One model call for several preprocessed inputs; outputs keep their order."""
        return [self.infer(inputs) for inputs in batch]

    @property
    def max_batch_size(self) -> int | None:
        """Largest batch infer_batch runs as one model call; None means no limit."""
        return 1

    def postprocess(self, outputs: Any, inputs: Any) -> list[Detection]:
        return cast("list[Detection]", outputs)

    def predict(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        inputs = self.preprocess(image)
        return self.postprocess(self.infer(inputs), inputs)

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
