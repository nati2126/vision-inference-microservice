"""YOLOv8 pre- and post-processing in plain NumPy.

The exported backends (ONNX Runtime, OpenVINO, TensorRT) run only the network
itself. Everything around it lives here, with no dependency on ultralytics:

    BGR image ──letterbox──▶ 640x640 ──normalise, HWC→CHW──▶ (1, 3, 640, 640)
                                                                │ network
    detections ◀──rescale── NMS ◀──confidence filter── decode ◀─┘ (1, 84, 8400)

Sharing one implementation across backends means accuracy differences in the
benchmark come from the runtimes and precisions, not from three slightly
different copies of this code.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from app.models.base import Detection

# Grey used by ultralytics for padding. Matching it keeps the exported models
# seeing the same border pixels they were trained and validated with.
PAD_VALUE = 114

# NMS is made class-aware by shifting each class's boxes into its own region
# of coordinate space, so boxes of different classes can never overlap. The
# offset only needs to exceed the largest possible coordinate.
_CLASS_OFFSET = 7680.0

# Cap on candidates entering NMS; bounds the worst case at very low
# confidence thresholds (the mAP evaluation runs at 0.001).
_MAX_NMS_CANDIDATES = 30000


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    """How an image was mapped into the network input, needed to undo it.

    A point ``(x, y)`` in the original image lands at
    ``(x * scale + pad_left, y * scale + pad_top)`` in the network input.
    """

    scale: float
    pad_left: int
    pad_top: int


def letterbox(
    image: npt.NDArray[np.uint8], size: int = 640
) -> tuple[npt.NDArray[np.uint8], LetterboxInfo]:
    """Resize ``image`` to fit a ``size x size`` square, preserving aspect ratio.

    A plain resize to 640x640 would squash a 4:3 photo and distort every
    object. Instead the image is scaled by a single factor so its longer side
    becomes ``size``, then the shorter side is padded equally on both sides.

    Args:
        image: ``(H, W, 3)`` image.
        size: Side of the square network input.

    Returns:
        The padded ``(size, size, 3)`` image and the transform applied.
    """
    height, width = image.shape[:2]
    scale = min(size / height, size / width)

    new_width, new_height = round(width * scale), round(height * scale)
    resized = image
    if (new_width, new_height) != (width, height):
        resized = np.asarray(
            cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR),
            dtype=np.uint8,
        )

    # Split the padding across both sides. The ±0.1 nudge rounds an odd
    # total so that the extra pixel goes to the bottom/right, matching
    # ultralytics exactly.
    pad_w, pad_h = (size - new_width) / 2, (size - new_height) / 2
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return np.asarray(padded, dtype=np.uint8), LetterboxInfo(scale, left, top)


def preprocess(
    image: npt.NDArray[np.uint8], size: int = 640
) -> tuple[npt.NDArray[np.float32], LetterboxInfo]:
    """Turn a BGR image into the ``(1, 3, size, size)`` float tensor YOLOv8 expects.

    Steps: letterbox, BGR→RGB (OpenCV decodes BGR, the model was trained on
    RGB), scale pixels from ``[0, 255]`` to ``[0, 1]``, reorder HWC→CHW
    (channels-first is the layout the convolutions are exported with) and add
    a batch dimension.
    """
    padded, info = letterbox(image, size)
    rgb = padded[:, :, ::-1]
    chw = rgb.transpose(2, 0, 1)
    # ascontiguousarray: the slicing/transposing above only made a strided
    # view; runtimes need a dense buffer.
    tensor = np.ascontiguousarray(chw, dtype=np.float32)[np.newaxis] / np.float32(255.0)
    return tensor, info


def nms(
    boxes: npt.NDArray[np.float32], scores: npt.NDArray[np.float32], iou_threshold: float
) -> npt.NDArray[np.intp]:
    """Greedy non-maximum suppression.

    Repeatedly keep the highest-scoring remaining box and discard every other
    box that overlaps it by more than ``iou_threshold``. The network predicts
    many near-duplicate boxes around each object; this collapses them to one.

    Args:
        boxes: ``(N, 4)`` boxes in xyxy format.
        scores: ``(N,)`` confidence of each box.
        iou_threshold: Overlap above which a box is suppressed.

    Returns:
        Indices of the kept boxes, highest score first.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]

        # Intersection of the best box with every remaining box, vectorised.
        inter_w = (np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest])).clip(min=0)
        inter_h = (np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest])).clip(min=0)
        inter = inter_w * inter_h
        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)

        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.intp)


def postprocess(
    output: npt.NDArray[np.float32],
    info: LetterboxInfo,
    original_shape: tuple[int, int],
    labels: list[str],
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    """Decode raw YOLOv8 output into detections on the original image.

    YOLOv8's head emits one ``(4 + num_classes)``-vector per anchor point:
    8400 of them for a 640 input (80x80 + 40x40 + 20x20 grid cells over the
    three strides). The first four values are the box centre and size
    ``(cx, cy, w, h)`` in network-input pixels; the rest are per-class
    scores, already passed through a sigmoid. Unlike YOLOv5 there is no
    separate objectness score, so a box's confidence is just its best class
    score.

    Args:
        output: Raw network output of shape ``(1, 4 + C, N)``.
        info: Letterbox transform returned by :func:`preprocess`.
        original_shape: ``(height, width)`` of the image before letterboxing.
        labels: Class names indexed by class id.
        confidence_threshold: Minimum score to keep a box.
        iou_threshold: NMS overlap threshold.
        max_detections: Maximum detections to return.
    """
    # (1, 84, 8400) -> (8400, 84): one row per candidate box.
    predictions = output[0].T.astype(np.float32, copy=False)

    # ── Confidence filter ────────────────────────────────────
    class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    mask = scores > confidence_threshold
    if not mask.any():
        return []
    predictions, class_ids, scores = predictions[mask], class_ids[mask], scores[mask]

    if len(scores) > _MAX_NMS_CANDIDATES:
        top = scores.argsort()[::-1][:_MAX_NMS_CANDIDATES]
        predictions, class_ids, scores = predictions[top], class_ids[top], scores[top]

    # ── Decode cx, cy, w, h -> x1, y1, x2, y2 ────────────────
    cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # ── Class-aware NMS ──────────────────────────────────────
    # Offsetting each class into its own coordinate region lets one NMS pass
    # handle every class at once without a person box suppressing a dog box.
    offset_boxes = boxes + (class_ids.astype(np.float32) * _CLASS_OFFSET)[:, None]
    keep = nms(offset_boxes, scores, iou_threshold)[:max_detections]
    boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]

    # ── Undo the letterbox ───────────────────────────────────
    boxes = scale_boxes_to_original(boxes, info, original_shape)

    # .tolist() converts to native Python floats/ints in one C call.
    box_list: list[list[float]] = boxes.tolist()
    class_list: list[int] = class_ids.tolist()
    score_list: list[float] = scores.tolist()
    return [
        Detection(
            class_id=cls,
            label=labels[cls] if cls < len(labels) else str(cls),
            confidence=score,
            x_min=box[0],
            y_min=box[1],
            x_max=box[2],
            y_max=box[3],
        )
        for box, cls, score in zip(box_list, class_list, score_list, strict=True)
    ]


def scale_boxes_to_original(
    boxes: npt.NDArray[np.float32], info: LetterboxInfo, original_shape: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Map xyxy boxes from network-input pixels back to original-image pixels.

    Inverts :func:`letterbox`: subtract the padding, divide by the scale,
    then clip to the image so no box extends past its edges.
    """
    height, width = original_shape
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_left) / info.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - info.pad_top) / info.scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out
