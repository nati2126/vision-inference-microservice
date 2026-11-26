"""ONNX Runtime backend (CPU or CUDA, FP32 or INT8)."""

import contextlib
import importlib

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from app.core.logging import get_logger
from app.models.exported import ExportedBackend
from app.models.labels import parse_names
from app.models.onnx_utils import onnx_precision

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


class OnnxRuntimeBackend(ExportedBackend):
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

        self._warm_up()
        logger.info("model_loaded", backend=self.name, precision=self._precision)

    def unload(self) -> None:
        self._session = None
        logger.info("model_unloaded", backend=self.name)

    def _forward(self, tensor: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert self._session is not None
        (output,) = self._session.run(None, {self._input_name: tensor})
        return np.asarray(output, dtype=np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._session is not None
