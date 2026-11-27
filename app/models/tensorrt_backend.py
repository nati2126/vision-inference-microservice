"""TensorRT backend: a prebuilt engine on an NVIDIA GPU, with torch tensors as I/O buffers."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorrt as trt
import torch

from app.core.logging import get_logger
from app.models.base import Detection
from app.models.exported import ExportedBackend, PreparedImage
from app.models.gpu_processing import postprocess_gpu, preprocess_gpu
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
        # Pre- and postprocessing, apart from the engine's stream so they do not
        # wait behind another request's engine call.
        self._io_stream: torch.cuda.Stream | None = None
        self._input: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._input_name = ""
        self._dynamic_batch = False
        self._bound_batch = 0

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

        context = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if n not in inputs]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("Engine must have exactly one input and one output tensor.")
        self._input_name, output_name = inputs[0], outputs[0]

        input_shape = tuple(engine.get_tensor_shape(self._input_name))
        self._dynamic_batch = input_shape[0] == -1
        if self._dynamic_batch:
            # Size the buffers for the largest batch in the optimisation profile.
            input_shape = tuple(engine.get_tensor_profile_shape(self._input_name, 0)[2])
        context.set_input_shape(self._input_name, input_shape)
        output_shape = tuple(context.get_tensor_shape(output_name))

        self._input = torch.empty(
            input_shape,
            dtype=_TORCH_DTYPES[engine.get_tensor_dtype(self._input_name)],
            device="cuda",
        )
        self._output = torch.empty(
            output_shape, dtype=_TORCH_DTYPES[engine.get_tensor_dtype(output_name)], device="cuda"
        )
        context.set_tensor_address(self._input_name, self._input.data_ptr())
        context.set_tensor_address(output_name, self._output.data_ptr())
        self._max_batch = self._bound_batch = int(input_shape[0])

        self._input_size = int(self._input.shape[2])
        self._labels = parse_names(metadata.get("names"))
        self._precision = str(metadata.get("precision", "unknown"))
        self._engine, self._context = engine, context
        self._stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        self._io_stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]

        self._warm_up()
        logger.info("model_loaded", backend=self.name, precision=self._precision)

    def unload(self) -> None:
        self._context = None
        self._engine = None
        self._input = self._output = None
        self._stream = self._io_stream = None
        torch.cuda.empty_cache()
        logger.info("model_unloaded", backend=self.name)

    def preprocess(self, image: npt.NDArray[np.uint8]) -> PreparedImage:
        assert self._input is not None, "Model is not loaded. Call .load() first."
        assert self._io_stream is not None
        tensor, info = preprocess_gpu(image, self._input_size, self._input.dtype, self._io_stream)
        return PreparedImage(tensor, info, (image.shape[0], image.shape[1]))

    def _forward_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return self._run(tensors[0] if len(tensors) == 1 else torch.cat(tensors))

    def _forward(self, tensor: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert self._input is not None
        on_gpu = torch.from_numpy(tensor).to(self._input.device, self._input.dtype)
        return np.asarray(self._run(on_gpu).cpu().numpy(), dtype=np.float32)

    def _run(self, tensor: torch.Tensor) -> torch.Tensor:
        """One engine call; returns a float32 copy of the output, still on the GPU."""
        assert self._context is not None and self._stream is not None
        assert self._input is not None and self._output is not None
        n = tensor.shape[0]
        with torch.cuda.stream(self._stream):
            if self._dynamic_batch and n != self._bound_batch:
                self._context.set_input_shape(self._input_name, tuple(tensor.shape))
                self._bound_batch = n
            self._input[:n].copy_(tensor)
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed.")
            # A copy, never a view: the next call overwrites the engine's buffer,
            # possibly while this output is still being postprocessed.
            output = self._output[:n].to(torch.float32, copy=True)
        self._stream.synchronize()
        return output

    def postprocess(self, outputs: torch.Tensor, inputs: PreparedImage) -> list[Detection]:
        assert self._io_stream is not None
        return postprocess_gpu(
            outputs,
            inputs.letterbox,
            original_shape=inputs.original_shape,
            labels=self._labels,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
            stream=self._io_stream,
        )

    @property
    def is_loaded(self) -> bool:
        return self._context is not None
