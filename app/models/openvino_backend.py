"""OpenVINO CPU backend (IR or ONNX; QDQ models run as INT8)."""

import numpy as np
import numpy.typing as npt
import openvino as ov

from app.core.logging import get_logger
from app.models.exported import ExportedBackend
from app.models.labels import parse_names

logger = get_logger(__name__)

NAMES_RT_INFO_KEY = ["model_info", "names"]


class OpenVINOBackend(ExportedBackend):
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

        self._warm_up()
        logger.info("model_loaded", backend=self.name, precision=self._precision)

    def unload(self) -> None:
        self._request = None
        self._compiled = None
        logger.info("model_unloaded", backend=self.name)

    def _forward(self, tensor: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert self._request is not None
        self._request.infer({0: tensor})
        # The output tensor's buffer is reused by the next infer call.
        return np.array(self._request.get_output_tensor(0).data, dtype=np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._request is not None
