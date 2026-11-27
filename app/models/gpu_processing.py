"""The parts of processing.py's pipeline that are faster on the GPU.

Used by the TensorRT backend, which has the network on the GPU already.

- Preprocessing uploads the uint8 image (0.9 MB at 640x480) instead of a
  4.9 MB float32 tensor, and pads, flips and scales it there. The resize stays
  in OpenCV: it is cheap (under 0.1 ms for 640x480, 1 ms for 810x1080), and a
  torch bilinear resize differs from it by one grey level on ~9% of pixels,
  which moved an FP16 box enough to fail parity.
- Postprocessing filters the 8400 anchors on the GPU. NMS stays in NumPy:
  torchvision's batched_nms took 0.52 ms against 0.10 ms for the ~50
  candidates left at the default threshold, and only won past several hundred.

Both run on a stream the caller owns. A fresh stream per call, from torch's
pool of 32, made the caching allocator keep blocks for every one of them:
406 MB reserved for 7 MB in use after 200 requests.
"""

from functools import cache

import cv2
import numpy as np
import numpy.typing as npt
import torch

from app.models.base import Detection
from app.models.processing import (
    PAD_VALUE,
    LetterboxInfo,
    candidates_to_detections,
    letterbox_geometry,
)


@cache
def _unit_scale(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """uint8 -> value / 255, computed by NumPy as the CPU path does.

    CUDA divides by a scalar as a multiply by its reciprocal, one ulp away from
    NumPy for many values; the table keeps the input bit-identical.
    """
    table = np.arange(256, dtype=np.float32) / np.float32(255.0)
    return torch.from_numpy(table).to(device, dtype)


def preprocess_gpu(
    image: npt.NDArray[np.uint8], size: int, dtype: torch.dtype, stream: torch.cuda.Stream
) -> tuple[torch.Tensor, LetterboxInfo]:
    """BGR image -> (1, 3, size, size) RGB tensor in [0, 1] on device."""
    height, width = image.shape[:2]
    g = letterbox_geometry(height, width, size)
    if (g.new_width, g.new_height) != (width, height):
        image = np.asarray(
            cv2.resize(image, (g.new_width, g.new_height), interpolation=cv2.INTER_LINEAR),
            dtype=np.uint8,
        )
    device = stream.device
    with torch.cuda.stream(stream):
        pixels = torch.from_numpy(np.ascontiguousarray(image)).to(device)
        canvas = torch.full((size, size, 3), PAD_VALUE, dtype=torch.uint8, device=device)
        canvas[g.top : g.top + g.new_height, g.left : g.left + g.new_width] = pixels
        # HWC BGR uint8 -> NCHW RGB in [0, 1], through the lookup table.
        chw = canvas.flip(2).permute(2, 0, 1).unsqueeze(0)
        tensor = _unit_scale(dtype, device)[chw.int()].contiguous()
    stream.synchronize()
    return tensor, g.info


def postprocess_gpu(
    output: torch.Tensor,
    info: LetterboxInfo,
    original_shape: tuple[int, int],
    labels: list[str],
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    stream: torch.cuda.Stream,
) -> list[Detection]:
    """Decode a (1, 4 + C, N) float32 output, as processing.postprocess does.

    The class maximum and confidence filter run on the GPU over all N anchors;
    only the survivors (tens at the default threshold) are copied to the host,
    in one transfer, for the same NMS the other backends use.
    """
    with torch.cuda.stream(stream):
        predictions = output[0].T
        scores, class_ids = predictions[:, 4:].max(dim=1)
        packed = torch.cat([predictions[:, :4], scores[:, None], class_ids[:, None].float()], 1)
        # Boolean indexing and .cpu() both wait for this stream.
        survivors = packed[scores > confidence_threshold].cpu().numpy()
    return candidates_to_detections(
        survivors[:, :4],
        survivors[:, 5].astype(np.intp),
        survivors[:, 4],
        info,
        original_shape,
        labels,
        iou_threshold,
        max_detections,
    )
