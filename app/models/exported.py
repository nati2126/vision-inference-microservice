"""Base for backends that run an exported YOLOv8 network with the NumPy pipeline."""

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from app.models.base import Detection, InferenceBackend
from app.models.processing import LetterboxInfo, postprocess, preprocess


@dataclass(frozen=True, slots=True)
class PreparedImage:
    tensor: npt.NDArray[np.float32]
    letterbox: LetterboxInfo
    original_shape: tuple[int, int]


class ExportedBackend(InferenceBackend):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
    ) -> None:
        super().__init__(model_path, device, confidence_threshold, iou_threshold, max_detections)
        self._input_size = 640
        self._labels: list[str] = []
        self._precision = "unknown"
        self._max_batch: int | None = 1

    @abstractmethod
    def _forward(self, tensor: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

    def preprocess(self, image: npt.NDArray[np.uint8]) -> PreparedImage:
        tensor, info = preprocess(image, self._input_size)
        return PreparedImage(tensor, info, (image.shape[0], image.shape[1]))

    def infer(self, inputs: PreparedImage) -> npt.NDArray[np.float32]:
        return self.infer_batch([inputs])[0]

    def infer_batch(self, batch: list[PreparedImage]) -> list[npt.NDArray[np.float32]]:
        self._require_loaded()
        if self._max_batch is not None and len(batch) > self._max_batch:
            raise ValueError(
                f"batch of {len(batch)} exceeds this model's maximum of {self._max_batch}"
            )
        tensors = [inputs.tensor for inputs in batch]
        # A single image needs no 4.9 MB copy into a new batch array.
        outputs = self._forward(tensors[0] if len(tensors) == 1 else np.concatenate(tensors))
        return [outputs[i : i + 1] for i in range(len(batch))]

    @property
    def max_batch_size(self) -> int | None:
        return self._max_batch

    def postprocess(
        self, outputs: npt.NDArray[np.float32], inputs: PreparedImage
    ) -> list[Detection]:
        return postprocess(
            outputs,
            inputs.letterbox,
            original_shape=inputs.original_shape,
            labels=self._labels,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )

    def _warm_up(self) -> None:
        self.predict(np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8))

    @property
    def precision(self) -> str:
        return self._precision
