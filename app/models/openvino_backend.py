"""OpenVINO CPU backend (IR or ONNX; QDQ models run as INT8)."""

import numpy as np
import numpy.typing as npt
import openvino as ov

from app.core.logging import get_logger
from app.models.base import Detection, InferenceBackend
from app.models.labels import parse_names
from app.models.processing import postprocess, preprocess

logger = get_logger(__name__)

NAMES_RT_INFO_KEY = ["model_info", "names"]


class OpenVINOBackend(InferenceBackend):
    name = "openvino"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
    ) -> None:
        if device != "cpu":
            raise ValueError(f"openvino backend is configured for 'cpu' only, got {device!r}")
        super().__init__(model_path, device, confidence_threshold, iou_threshold, max_detections)
        self._request: ov.InferRequest | None = None
        self._compiled: ov.CompiledModel | None = None
        self._input_size = 640
        self._labels: list[str] = []
        self._precision = "unknown"

    def load(self) -> None:
        logger.info("loading_model", backend=self.name, path=self.model_path, device=self.device)
        core = ov.Core()
        model = core.read_model(self.model_path)

        is_quantized = any(op.get_type_name() == "FakeQuantize" for op in model.get_ops())
        names = (
            str(model.get_rt_info(NAMES_RT_INFO_KEY))
            if model.has_rt_info(NAMES_RT_INFO_KEY)
            else None
        )

        compiled = core.compile_model(
            model,
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                # Otherwise CPUs with AMX/AVX512-BF16 silently run FP32 models in bf16.
                "INFERENCE_PRECISION_HINT": "f32",
            },
        )

        height = model.input(0).get_partial_shape()[2]
        self._input_size = height.get_length() if height.is_static else 640
        self._labels = parse_names(names)
        executed = str(compiled.get_property("INFERENCE_PRECISION_HINT")).lower()
        self._precision = (
            "int8" if is_quantized else {"f16": "fp16", "bf16": "bf16"}.get(executed, "fp32")
        )
        self._compiled = compiled
        self._request = compiled.create_infer_request()

        self.predict(np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8))
        logger.info("model_loaded", backend=self.name, precision=self._precision)

    def unload(self) -> None:
        self._request = None
        self._compiled = None
        logger.info("model_unloaded", backend=self.name)

    def predict(self, image: npt.NDArray[np.uint8]) -> list[Detection]:
        if self._request is None:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        tensor, info = preprocess(image, self._input_size)
        self._request.infer({0: tensor})
        output = self._request.get_output_tensor(0).data
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
        return self._request is not None

    @property
    def precision(self) -> str:
        return self._precision
