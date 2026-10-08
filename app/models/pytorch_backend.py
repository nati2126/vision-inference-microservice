"""PyTorch backend: the original ultralytics pipeline, kept as the baseline.

Everything else in the benchmark is measured against this. It is also the
only backend that still uses ultralytics at inference time; its pre- and
post-processing are ultralytics' own, which is what makes it a useful
reference for the parity test against :mod:`app.models.processing`.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from ultralytics import YOLO

from app.core.logging import get_logger
from app.models.base import Detection, InferenceBackend

logger = get_logger(__name__)


class PyTorchBackend(InferenceBackend):
    """YOLOv8 through ``ultralytics.YOLO`` on CPU, CUDA or MPS.

    Args:
        weights_dir: Directory that a bare weight filename resolves against.
        input_size: Long side of the network input.
        (others): See :class:`InferenceBackend`.
    """

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
        # Hand ultralytics a ``torch.device``, never a string. Given the
        # string "cpu", ``select_device`` sets CUDA_VISIBLE_DEVICES=-1 for the
        # whole process and then queries CUDA, which caches "no GPU" for the
        # process lifetime, breaking any other CUDA user (ONNX Runtime,
        # TensorRT) in the same process. A ``torch.device`` short-circuits it.
        self._torch_device = torch.device("cuda:0" if device == "cuda" else device)

    # ── Lifecycle ────────────────────────────────────────────

    def _resolve_weights(self) -> str:
        """Return the path to hand to ultralytics.

        A bare filename (``yolov8n.pt``) is resolved against ``weights_dir``.
        Ultralytics downloads a *relative* filename into the current working
        directory; in the container that is the root-owned ``/app``, so an
        unprivileged process gets ``PermissionError`` on first start. An
        absolute path avoids that and lets the weights persist on a volume.

        An explicit path in ``model_path`` (absolute, or containing a
        separator) is passed through untouched.
        """
        candidate = Path(self.model_path)
        if self.weights_dir is None or candidate.is_absolute() or candidate.parent != Path("."):
            return self.model_path

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        return str(self.weights_dir / candidate.name)

    def load(self) -> None:
        """Load model weights and run one warm-up inference."""
        weights = self._resolve_weights()
        logger.info("loading_model", backend=self.name, weights_path=weights, device=self.device)
        self._model = YOLO(weights)
        # Warm-up: the first call pays for CUDA context creation, cuDNN
        # autotuning and lazy allocations. Paying it here keeps it off the
        # first real request.
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.predict(dummy)
        logger.info("model_loaded", backend=self.name, precision=self.precision)

    def unload(self) -> None:
        """Release model resources."""
        self._model = None
        logger.info("model_unloaded", backend=self.name)

    # ── Inference ────────────────────────────────────────────

    def predict(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        """Run detection on a BGR ``(H, W, 3)`` image."""
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
            # Read the batched tensors once rather than iterating ``Boxes``:
            # fewer host/device round-trips, and ``Boxes`` is not typed as
            # iterable so strict mypy rejects the per-box loop.
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
        """Check whether the model weights are currently in memory."""
        return self._model is not None

    @property
    def precision(self) -> str:
        """Ultralytics runs ``.pt`` weights in FP32 unless ``half=True`` is passed."""
        return "fp32"
