"""COCO-style mAP50 and mAP50-95 (greedy matching, 101-point interpolation)."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_RECALL_LEVELS = np.linspace(0.0, 1.0, 101)


def box_iou(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` xyxy boxes -> ``(N, M)``."""
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod((bottom_right - top_left).clip(min=0), axis=2)
    area_a = np.prod((a[:, 2:] - a[:, :2]).clip(min=0), axis=1)
    area_b = np.prod((b[:, 2:] - b[:, :2]).clip(min=0), axis=1)
    return np.asarray(inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9), dtype=np.float32)


def average_precision(
    scores: npt.NDArray[np.float32], true_positive: npt.NDArray[np.bool_], num_gt: int
) -> float:
    """101-point interpolated AP for one class at one IoU threshold."""
    if num_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(true_positive[order])
    fp = np.cumsum(~true_positive[order])
    recall = tp / num_gt
    precision = tp / (tp + fp)

    precision = np.maximum.accumulate(precision[::-1])[::-1]

    idx = np.searchsorted(recall, _RECALL_LEVELS, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


@dataclass
class DetectionEvaluator:
    """Accumulates per-image results, then reports mAP50 and mAP50-95."""

    _scores: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    _hits: dict[int, list[npt.NDArray[np.bool_]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _num_gt: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    num_images: int = 0

    def add(
        self,
        pred_boxes: npt.NDArray[np.float32],
        pred_scores: npt.NDArray[np.float32],
        pred_classes: npt.NDArray[np.int64],
        gt_boxes: npt.NDArray[np.float32],
        gt_classes: npt.NDArray[np.int64],
    ) -> None:
        """Match one image's predictions against its ground truth."""
        self.num_images += 1
        for cls in np.unique(gt_classes):
            self._num_gt[int(cls)] += int((gt_classes == cls).sum())

        for cls in np.unique(pred_classes):
            pred_mask = pred_classes == cls
            boxes, scores = pred_boxes[pred_mask], pred_scores[pred_mask]
            gts = gt_boxes[gt_classes == cls]

            hits = np.zeros((len(boxes), len(IOU_THRESHOLDS)), dtype=bool)
            if len(gts):
                iou = box_iou(boxes, gts)
                order = np.argsort(-scores, kind="stable")
                for t_index, threshold in enumerate(IOU_THRESHOLDS):
                    claimed = np.zeros(len(gts), dtype=bool)
                    for p in order:
                        candidates = np.where(claimed, -1.0, iou[p])
                        best = int(candidates.argmax())
                        if candidates[best] >= threshold:
                            claimed[best] = True
                            hits[p, t_index] = True

            self._scores[int(cls)].extend(scores.tolist())
            self._hits[int(cls)].extend(hits)

    def summarize(self) -> dict[str, float]:
        """Return ``map50`` and ``map50_95`` over classes present in the ground truth."""
        per_class = []
        for cls, num_gt in self._num_gt.items():
            scores = np.asarray(self._scores.get(cls, []), dtype=np.float32)
            hits = (
                np.stack(self._hits[cls])
                if self._hits.get(cls)
                else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
            )
            per_class.append(
                [average_precision(scores, hits[:, t], num_gt) for t in range(len(IOU_THRESHOLDS))]
            )

        ap = np.asarray(per_class)  # (classes, thresholds)
        return {
            "map50": float(ap[:, 0].mean()),
            "map50_95": float(ap.mean()),
            "num_images": self.num_images,
            "num_classes": len(per_class),
            "num_gt_boxes": int(sum(self._num_gt.values())),
        }
