"""The GPU pre/post-processing gives what the NumPy pipeline gives."""

import numpy as np
import numpy.typing as npt
import pytest
import torch

from app.models.gpu_processing import postprocess_gpu, preprocess_gpu
from app.models.labels import COCO_CLASSES
from app.models.processing import LetterboxInfo, postprocess, preprocess

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@pytest.fixture(scope="module")
def stream() -> torch.cuda.Stream:
    return torch.cuda.Stream()  # type: ignore[no-untyped-call]


@pytest.mark.parametrize(
    ("height", "width"), [(480, 640), (1080, 810), (640, 640), (100, 37), (2000, 3000)]
)
def test_preprocess_is_bit_identical_to_numpy(
    height: int, width: int, stream: torch.cuda.Stream
) -> None:
    image = np.random.default_rng(height * width).integers(0, 256, (height, width, 3), np.uint8)
    want, want_info = preprocess(image, 640)

    got, info = preprocess_gpu(image, 640, torch.float32, stream)

    assert info == want_info
    assert got.shape == want.shape
    assert np.array_equal(got.cpu().numpy(), want)


def test_preprocess_casts_to_the_engine_dtype(stream: torch.cuda.Stream) -> None:
    image = np.random.default_rng(0).integers(0, 256, (480, 640, 3), np.uint8)
    want, _ = preprocess(image, 640)

    got, _ = preprocess_gpu(image, 640, torch.float16, stream)

    assert got.dtype == torch.float16
    assert np.array_equal(got.cpu().numpy(), want.astype(np.float16))


def _synthetic_output(n_boxes: int, seed: int = 0) -> npt.NDArray[np.float32]:
    """(1, 84, 8400) with n_boxes confident, overlapping boxes among background noise."""
    rng = np.random.default_rng(seed)
    output = np.zeros((1, 84, 8400), np.float32)
    output[0, 4:] = rng.uniform(0, 0.2, (80, 8400))
    output[0, :2] = rng.uniform(50, 590, (2, 8400))
    output[0, 2:4] = rng.uniform(10, 200, (2, 8400))
    hits = rng.choice(8400, n_boxes, replace=False)
    output[0, 4 + rng.integers(0, 80, n_boxes), hits] = rng.uniform(0.3, 0.99, n_boxes)
    # Slightly weaker near-duplicates of the first few boxes, for NMS to remove.
    copied = hits[:10]
    dupes = rng.choice(np.setdiff1d(np.arange(8400), hits), len(copied), replace=False)
    output[:, :, dupes] = output[:, :, copied]
    output[0, :2, dupes] += 2.0
    output[0, 4:, dupes] *= 0.9
    return output


@pytest.mark.parametrize(("n_boxes", "max_detections"), [(0, 300), (40, 300), (400, 300), (40, 5)])
def test_postprocess_matches_numpy(
    n_boxes: int, max_detections: int, stream: torch.cuda.Stream
) -> None:
    output = _synthetic_output(n_boxes)
    info = LetterboxInfo(scale=0.5926, pad_left=80, pad_top=0)
    kwargs = {
        "info": info,
        "original_shape": (1080, 810),
        "labels": COCO_CLASSES,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.7,
        "max_detections": max_detections,
    }
    want = postprocess(output, **kwargs)  # type: ignore[arg-type]

    got = postprocess_gpu(torch.from_numpy(output).cuda(), stream=stream, **kwargs)  # type: ignore[arg-type]

    assert len(got) == len(want) == min(n_boxes, max_detections)
    for g, w in zip(got, want, strict=True):
        assert (g.class_id, g.label) == (w.class_id, w.label)
        assert g.confidence == pytest.approx(w.confidence, abs=1e-6)
        assert (g.x_min, g.y_min, g.x_max, g.y_max) == pytest.approx(
            (w.x_min, w.y_min, w.x_max, w.y_max), abs=1e-3
        )
