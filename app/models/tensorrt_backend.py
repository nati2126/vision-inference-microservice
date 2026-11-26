"""TensorRT backend: a prebuilt engine on an NVIDIA GPU, with torch tensors as I/O buffers."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorrt as trt
import torch

from app.core.logging import get_logger
from app.models.exported import ExportedBackend
from app.models.labels import parse_names

logger = get_logger(__name__)

_TORCH_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
}


def engine_metadata_path(engine_path: str | Path) -> Path:
    return Path(f"{engine_path}.json")


class TensorRTBackend(ExportedBackend):
    name = "tensorrt"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
    ) -> None:
        if device != "cuda":
            raise ValueError(f"tensorrt backend requires device 'cuda', got {device!r}")
        super().__init__(model_path, device, confidence_threshold, iou_threshold, max_detections)
        self._engine: Any = None
        self._context: Any = None
        self._stream: torch.cuda.Stream | None = None
        self._input: torch.Tensor | None = None
        self._output: torch.Tensor | None = None

    def load(self) -> None:
        logger.info("loading_model", backend=self.name, path=self.model_path, device=self.device)
        if not torch.cuda.is_available():
            raise RuntimeError("tensorrt backend needs a CUDA device, none is available.")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(Path(self.model_path).read_bytes())
        if engine is None:
            # Usually a GPU or TensorRT version mismatch; rebuild on this machine.
            raise RuntimeError(f"Could not deserialize TensorRT engine {self.model_path!r}.")

        metadata: dict[str, Any] = {}
        sidecar = engine_metadata_path(self.model_path)
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))

        buffers: dict[str, torch.Tensor] = {}
        context = engine.create_execution_context()
        for index in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(index)
            shape = tuple(engine.get_tensor_shape(tensor_name))
            dtype = _TORCH_DTYPES[engine.get_tensor_dtype(tensor_name)]
            buffers[tensor_name] = torch.empty(shape, dtype=dtype, device="cuda")
            context.set_tensor_address(tensor_name, buffers[tensor_name].data_ptr())
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self._input = buffers[tensor_name]
            else:
                self._output = buffers[tensor_name]

        if self._input is None or self._output is None:
            raise RuntimeError("Engine must have exactly one input and one output tensor.")

        self._input_size = int(self._input.shape[2])
        self._labels = parse_names(metadata.get("names"))
        self._precision = str(metadata.get("precision", "unknown"))
        self._engine, self._context = engine, context
        self._stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]

        self._warm_up()
        logger.info("model_loaded", backend=self.name, precision=self._precision)

    def unload(self) -> None:
        self._context = None
        self._engine = None
        self._input = self._output = None
        self._stream = None
        torch.cuda.empty_cache()
        logger.info("model_unloaded", backend=self.name)

    def _forward(self, tensor: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert self._context is not None and self._stream is not None
        assert self._input is not None and self._output is not None
        with torch.cuda.stream(self._stream):
            self._input.copy_(torch.from_numpy(tensor).to(self._input.dtype))
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed.")
            output = self._output.float().cpu()
        self._stream.synchronize()
        return output.numpy()

    @property
    def is_loaded(self) -> bool:
        return self._context is not None
