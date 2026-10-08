"""NumPy pre/post-processing."""

import numpy as np
import pytest

from app.models.labels import COCO_CLASSES
from app.models.processing import (
    PAD_VALUE,
    LetterboxInfo,
    letterbox,
    nms,
    postprocess,
    preprocess,
    scale_boxes_to_original,
)


def _to_letterboxed(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = out[:, [0, 2]] * info.scale + info.pad_left
    out[:, [1, 3]] = out[:, [1, 3]] * info.scale + info.pad_top
    return out


@pytest.mark.parametrize(("height", "width"), [(480, 640), (1080, 810), (640, 640), (100, 37)])
def test_letterbox_output_is_square_and_keeps_aspect(height: int, width: int) -> None:
    image = np.full((height, width, 3), 200, dtype=np.uint8)

    padded, info = letterbox(image, 640)

    assert padded.shape == (640, 640, 3)
    assert padded.dtype == np.uint8
    assert info.scale == pytest.approx(640 / max(height, width))
    content_w, content_h = round(width * info.scale), round(height * info.scale)
    assert abs(2 * info.pad_left + content_w - 640) <= 1
    assert abs(2 * info.pad_top + content_h - 640) <= 1
    if info.pad_top > 0:
        assert (padded[: info.pad_top] == PAD_VALUE).all()
    if info.pad_left > 0:
        assert (padded[:, : info.pad_left] == PAD_VALUE).all()


@pytest.mark.parametrize(("height", "width"), [(480, 640), (1080, 810), (333, 1000)])
def test_letterbox_then_rescale_round_trips_boxes(height: int, width: int) -> None:
    _, info = letterbox(np.zeros((height, width, 3), dtype=np.uint8), 640)
    boxes = np.array(
        [
            [0, 0, width, height],
            [10.5, 20.25, 100.0, 200.0],
            [width / 3, height / 4, width / 2, height / 2],
        ],
        dtype=np.float32,
    )

    recovered = scale_boxes_to_original(_to_letterboxed(boxes, info), info, (height, width))

    np.testing.assert_allclose(recovered, boxes, atol=1e-3)


def test_rescale_clips_boxes_to_the_image() -> None:
    info = LetterboxInfo(scale=1.0, pad_left=0, pad_top=80)
    boxes = np.array([[-5, 70, 700, 600]], dtype=np.float32)

    out = scale_boxes_to_original(boxes, info, (480, 640))

    np.testing.assert_allclose(out, [[0, 0, 640, 480]])


def test_preprocess_layout_dtype_and_range() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 2] = 255

    tensor, _ = preprocess(image, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert tensor[0, 0, 320, 320] == pytest.approx(1.0)
    assert tensor[0, 2, 320, 320] == pytest.approx(0.0)


def test_nms_suppresses_overlapping_boxes_keeping_the_best() -> None:
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [5, 5, 105, 105],
            [200, 200, 300, 300],
            [0, 0, 100, 50],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)

    keep = nms(boxes, scores, iou_threshold=0.7)

    assert keep.tolist() == [0, 2, 3]


def test_nms_returns_indices_in_descending_score_order() -> None:
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.2, 0.9, 0.5], dtype=np.float32)

    assert nms(boxes, scores, 0.5).tolist() == [1, 2, 0]


def test_nms_handles_empty_input() -> None:
    keep = nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.5)

    assert keep.shape == (0,)


def _raw_output(candidates: list[tuple[float, float, float, float, int, float]]) -> np.ndarray:
    out = np.zeros((1, 4 + len(COCO_CLASSES), len(candidates)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(candidates):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_postprocess_decodes_filters_and_rescales() -> None:
    info = LetterboxInfo(scale=1.0, pad_left=0, pad_top=80)
    output = _raw_output(
        [
            (100, 180, 40, 60, 0, 0.9),
            (102, 181, 40, 60, 0, 0.8),
            (102, 181, 40, 60, 16, 0.7),
            (400, 300, 50, 50, 2, 0.1),
        ]
    )

    detections = postprocess(output, info, (480, 640), COCO_CLASSES, 0.25, 0.7, 300)

    assert [(d.label, round(d.confidence, 2)) for d in detections] == [
        ("person", 0.9),
        ("dog", 0.7),
    ]
    person = detections[0]
    assert (person.x_min, person.y_min, person.x_max, person.y_max) == pytest.approx(
        (80, 70, 120, 130)
    )


def test_postprocess_respects_max_detections() -> None:
    output = _raw_output([(i * 30 + 15, 300, 20, 20, 0, 0.5 + i / 100) for i in range(20)])

    detections = postprocess(
        output, LetterboxInfo(1.0, 0, 0), (640, 640), COCO_CLASSES, 0.25, 0.7, 5
    )

    assert len(detections) == 5
    assert detections[0].confidence == pytest.approx(0.69)


def test_postprocess_with_nothing_above_threshold_is_empty() -> None:
    output = _raw_output([(100, 100, 10, 10, 0, 0.1)])

    assert (
        postprocess(output, LetterboxInfo(1.0, 0, 0), (640, 640), COCO_CLASSES, 0.25, 0.7, 300)
        == []
    )
