"""Build the configured inference backend.

Backend modules are imported lazily: OpenVINO and TensorRT are optional
extras, and a missing package should only matter when that backend is
actually selected.
"""

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
    """Instantiate (but do not load) the backend called ``backend``.

    Raises:
        ValueError: For an unknown backend name or an unsupported device.
        ImportError: If the backend's optional dependency is not installed.
    """
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
