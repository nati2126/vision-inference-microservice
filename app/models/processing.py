"""YOLOv8 pre- and post-processing in NumPy, shared by the exported backends."""

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from app.models.base import Detection

PAD_VALUE = 114  # ultralytics' letterbox grey
_CLASS_OFFSET = 7680.0  # per-class coordinate shift for class-aware NMS
MAX_NMS_CANDIDATES = 30000


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    """Maps original (x, y) to (x * scale + pad_left, y * scale + pad_top)."""

    scale: float
    pad_left: int
    pad_top: int


@dataclass(frozen=True, slots=True)
class LetterboxGeometry:
    new_width: int
    new_height: int
    top: int
    bottom: int
    left: int
    right: int
    info: LetterboxInfo


def letterbox_geometry(height: int, width: int, size: int) -> LetterboxGeometry:
    """Resized size and padding that fit an image into a size x size square."""
    scale = min(size / height, size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    # The ±0.1 sends an odd pixel of padding to the bottom/right, as ultralytics does.
    pad_w, pad_h = (size - new_width) / 2, (size - new_height) / 2
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    return LetterboxGeometry(
        new_width, new_height, top, bottom, left, right, LetterboxInfo(scale, left, top)
    )


def letterbox(
    image: npt.NDArray[np.uint8], size: int = 640
) -> tuple[npt.NDArray[np.uint8], LetterboxInfo]:
    """Resize to fit a size x size square keeping aspect ratio, then pad."""
    height, width = image.shape[:2]
    g = letterbox_geometry(height, width, size)

    resized = image
    if (g.new_width, g.new_height) != (width, height):
        resized = np.asarray(
            cv2.resize(image, (g.new_width, g.new_height), interpolation=cv2.INTER_LINEAR),
            dtype=np.uint8,
        )

    padded = cv2.copyMakeBorder(
        resized, g.top, g.bottom, g.left, g.right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return np.asarray(padded, dtype=np.uint8), g.info


def preprocess(
    image: npt.NDArray[np.uint8], size: int = 640
) -> tuple[npt.NDArray[np.float32], LetterboxInfo]:
    """BGR image -> (1, 3, size, size) RGB float tensor in [0, 1]."""
    padded, info = letterbox(image, size)
    chw = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.asarray(
        np.ascontiguousarray(chw, dtype=np.float32)[np.newaxis] / np.float32(255.0),
        dtype=np.float32,
    )
    return tensor, info


def nms(
    boxes: npt.NDArray[np.float32], scores: npt.NDArray[np.float32], iou_threshold: float
) -> npt.NDArray[np.intp]:
    """Greedy NMS over xyxy boxes; returns kept indices, highest score first."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]

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
    """Decode (1, 4 + C, N) YOLOv8 output into detections on the original image.

    Rows are (cx, cy, w, h) followed by per-class sigmoid scores; YOLOv8 has
    no objectness, so confidence is the best class score.
    """
    predictions = output[0].T.astype(np.float32, copy=False)

    class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    mask = scores > confidence_threshold
    return candidates_to_detections(
        predictions[mask, :4],
        class_ids[mask],
        scores[mask],
        info,
        original_shape,
        labels,
        iou_threshold,
        max_detections,
    )


def candidates_to_detections(
    predictions: npt.NDArray[np.float32],
    class_ids: npt.NDArray[np.intp],
    scores: npt.NDArray[np.float32],
    info: LetterboxInfo,
    original_shape: tuple[int, int],
    labels: list[str],
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    """NMS and rescaling for the (cx, cy, w, h) boxes that passed the confidence threshold."""
    if len(scores) == 0:
        return []
    if len(scores) > MAX_NMS_CANDIDATES:
        top = scores.argsort()[::-1][:MAX_NMS_CANDIDATES]
        predictions, class_ids, scores = predictions[top], class_ids[top], scores[top]

    cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    boxes = np.asarray(
        np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1), dtype=np.float32
    )

    offset_boxes = boxes + (class_ids.astype(np.float32) * _CLASS_OFFSET)[:, None]
    keep = nms(offset_boxes, scores, iou_threshold)[:max_detections]
    boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]

    boxes = scale_boxes_to_original(boxes, info, original_shape)
    return to_detections(boxes.tolist(), class_ids.tolist(), scores.tolist(), labels)


def to_detections(
    box_list: list[list[float]], class_list: list[int], score_list: list[float], labels: list[str]
) -> list[Detection]:
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
    """Invert the letterbox and clip xyxy boxes to the original image."""
    height, width = original_shape
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_left) / info.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - info.pad_top) / info.scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out
