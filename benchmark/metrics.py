"""COCO-style mean average precision, written out so it can be explained.

Every backend is scored by this one function with identical inputs, so any
difference in the numbers comes from the backend, not the evaluation.

The procedure, for each IoU threshold t in 0.50, 0.55, ..., 0.95:

1. Matching (per image, per class): walk the predictions from most to
   least confident. Each one claims the not-yet-claimed ground-truth box it
   overlaps most; if that overlap is at least t it is a true positive,
   otherwise a false positive. A ground-truth box can be claimed only once,
   so duplicate detections of one object count against the model.
2. Precision-recall curve (per class, over all images): sort every
   prediction by confidence and accumulate TPs and FPs. Each prefix of the
   list is one operating point: precision = TP / (TP + FP),
   recall = TP / number of ground-truth boxes.
3. Average precision: make precision monotonically non-increasing in
   recall (the "envelope"), sample it at 101 recall levels 0.00 .. 1.00 and
   average, as pycocotools does.

mAP50 is the mean AP over classes at t = 0.50; mAP50-95 also averages over
the ten thresholds. Classes with no ground truth in the dataset are skipped.
"""

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

    # Envelope: the best precision achievable at this recall *or higher*.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # For each recall level r, the precision at the first point reaching r.
    idx = np.searchsorted(recall, _RECALL_LEVELS, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


@dataclass
class DetectionEvaluator:
    """Accumulates per-image results, then reports mAP50 and mAP50-95."""

    # class id -> list of (score, tp flags at each IoU threshold)
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
