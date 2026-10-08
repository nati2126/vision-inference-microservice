"""ONNX Runtime backend (CPU or CUDA, FP32 or INT8)."""

import contextlib
import importlib

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from app.core.logging import get_logger
from app.models.base import Detection, InferenceBackend
from app.models.labels import parse_names
from app.models.onnx_utils import onnx_precision
from app.models.processing import postprocess, preprocess

logger = get_logger(__name__)

_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


def _preload_cuda_libraries() -> None:
    # The CUDA provider needs cuBLAS/cuDNN; importing torch loads its bundled copies.
    try:
        importlib.import_module("torch")
    except ImportError:
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            with contextlib.suppress(Exception):
                preload()


class OnnxRuntimeBackend(InferenceBackend):
    name = "onnxruntime"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
    ) -> None:
        if device not in _PROVIDERS:
            raise ValueError(f"onnxruntime backend supports {sorted(_PROVIDERS)}, got {device!r}")
        super().__init__(model_path, device, confidence_threshold, iou_threshold, max_detections)
        self._session: ort.InferenceSession | None = None
        self._input_name = ""
        self._input_size = 640
        self._labels: list[str] = []
        self._precision = "unknown"

    def load(self) -> None:
        logger.info("loading_model", backend=self.name, path=self.model_path, device=self.device)
        if self.device == "cuda":
            _preload_cuda_libraries()

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            self.model_path, sess_options=options, providers=_PROVIDERS[self.device]
        )

        # ORT silently falls back to CPU when a provider fails to initialise.
        requested = _PROVIDERS[self.device][0]
        if session.get_providers()[0] != requested:
            raise RuntimeError(
                f"{requested} was requested but the session runs on "
                f"{session.get_providers()}; check the CUDA / cuDNN installation."
            )

        model_input = session.get_inputs()[0]
        self._input_name = model_input.name
        height = model_input.shape[2]
        self._input_size = height if isinstance(height, int) else 640
        self._labels = parse_names(session.get_modelmeta().custom_metadata_map.get("names"))
        self._precision = onnx_precision(self.model_path)
        self._session = session

        self.predict(np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8))
        logger.info("model_loaded", backend=self.name, precision=self._precision)

    def unload(self) -> None:
        self._session = None
        logger.info("model_unloaded", backend=self.name)

    def predict(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        if self._session is None:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        tensor, info = preprocess(image, self._input_size)
        (output,) = self._session.run(None, {self._input_name: tensor})
        return postprocess(
            np.asarray(output),
            info,
            original_shape=(image.shape[0], image.shape[1]),
            labels=self._labels,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def precision(self) -> str:
        return self._precision
