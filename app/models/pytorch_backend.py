"""PyTorch backend via ultralytics; the baseline the other backends are compared to."""

import os
from pathlib import Path

# Stop ultralytics from pip-installing packages at runtime. Must precede its import.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
import numpy.typing as npt
import torch
from ultralytics import YOLO

from app.core.logging import get_logger
from app.models.base import Detection, InferenceBackend

logger = get_logger(__name__)


class PyTorchBackend(InferenceBackend):
    name = "pytorch"

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        weights_dir: Path | None = None,
        input_size: int = 640,
    ) -> None:
        super().__init__(model_path, device, confidence_threshold, iou_threshold, max_detections)
        self.weights_dir = weights_dir
        self.input_size = input_size
        self._model: YOLO | None = None
        # A "cpu" string makes ultralytics set CUDA_VISIBLE_DEVICES=-1 process-wide.
        self._torch_device = torch.device("cuda:0" if device == "cuda" else device)

    def _resolve_weights(self) -> str:
        # Bare filenames go to weights_dir; ultralytics would otherwise download into the CWD.
        candidate = Path(self.model_path)
        if self.weights_dir is None or candidate.is_absolute() or candidate.parent != Path("."):
            return self.model_path

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        return str(self.weights_dir / candidate.name)

    def load(self) -> None:
        weights = self._resolve_weights()
        logger.info("loading_model", backend=self.name, weights_path=weights, device=self.device)
        self._model = YOLO(weights)
        self.predict(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
        logger.info("model_loaded", backend=self.name, precision=self.precision)

    def unload(self) -> None:
        self._model = None
        logger.info("model_unloaded", backend=self.name)

    def predict(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            imgsz=self.input_size,
            device=self._torch_device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            coords: list[list[float]] = boxes.xyxy.tolist()
            confidences: list[float] = boxes.conf.tolist()
            class_ids: list[float] = boxes.cls.tolist()

            for xyxy, confidence, class_id in zip(coords, confidences, class_ids, strict=True):
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        label=str(result.names[int(class_id)]),
                        confidence=float(confidence),
                        x_min=float(xyxy[0]),
                        y_min=float(xyxy[1]),
                        x_max=float(xyxy[2]),
                        y_max=float(xyxy[3]),
                    )
                )

        return detections

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def precision(self) -> str:
        return "fp32"
