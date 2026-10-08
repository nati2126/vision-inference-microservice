"""Backend factory; imports lazily so optional extras are only needed when selected."""

from pathlib import Path
from typing import Literal

from app.models.base import InferenceBackend

BackendName = Literal["pytorch", "onnxruntime", "openvino", "tensorrt"]
BACKENDS: tuple[BackendName, ...] = ("pytorch", "onnxruntime", "openvino", "tensorrt")


def create_backend(
    backend: str,
    model_path: str,
    device: str = "cpu",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
    weights_dir: Path | None = None,
) -> InferenceBackend:
    backend_cls: type[InferenceBackend]
    if backend == "pytorch":
        from app.models.pytorch_backend import PyTorchBackend

        return PyTorchBackend(
            model_path,
            device,
            confidence_threshold,
            iou_threshold,
            max_detections,
            weights_dir=weights_dir,
        )
    if backend == "onnxruntime":
        from app.models.onnxruntime_backend import OnnxRuntimeBackend

        backend_cls = OnnxRuntimeBackend
    elif backend == "openvino":
        from app.models.openvino_backend import OpenVINOBackend

        backend_cls = OpenVINOBackend
    elif backend == "tensorrt":
        from app.models.tensorrt_backend import TensorRTBackend

        backend_cls = TensorRTBackend
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}.")

    return backend_cls(model_path, device, confidence_threshold, iou_threshold, max_detections)
