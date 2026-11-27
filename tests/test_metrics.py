"""The in-house mAP against pycocotools, the reference implementation."""

import contextlib
import io

import numpy as np
import numpy.typing as npt
import pytest

from benchmark.metrics import DetectionEvaluator, average_precision

pycocotools = pytest.importorskip("pycocotools")


def _scene(
    rng: np.random.Generator, n_gt: int
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int64],
]:
    """Ground truth plus predictions: jittered hits, misses, duplicates and false positives."""
    xy = rng.uniform(0, 500, (n_gt, 2))
    gt = np.concatenate([xy, xy + rng.uniform(20, 140, (n_gt, 2))], axis=1).astype(np.float32)
    gt_cls = rng.integers(0, 4, n_gt)

    found = rng.random(n_gt) < 0.8
    jitter = rng.normal(0, 6, (n_gt, 4))
    pred = [gt[found] + jitter[found]]
    pred_cls = [gt_cls[found]]
    dupes = rng.random(n_gt) < 0.2
    pred.append(gt[dupes] + rng.normal(0, 12, (int(dupes.sum()), 4)))
    pred_cls.append(gt_cls[dupes])
    n_fp = int(rng.integers(0, 6))
    fp_xy = rng.uniform(0, 500, (n_fp, 2))
    pred.append(np.concatenate([fp_xy, fp_xy + rng.uniform(20, 140, (n_fp, 2))], axis=1))
    pred_cls.append(rng.integers(0, 4, n_fp))

    boxes = np.concatenate(pred).astype(np.float32)
    # Jitter can flip a box inside out. A detector never outputs one, and the two
    # evaluators treat it differently (pycocotools ignores negative areas).
    boxes[:, 2:] = np.maximum(boxes[:, 2:], boxes[:, :2] + 1.0)
    classes = np.concatenate(pred_cls).astype(np.int64)
    scores = rng.uniform(0.05, 1.0, len(boxes)).astype(np.float32)
    return gt, gt_cls.astype(np.int64), boxes, scores, classes


def _coco_map(scenes: list[tuple[npt.NDArray[np.float32], ...]]) -> tuple[float, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, annotations, detections = [], [], []
    for image_id, (gt, gt_cls, boxes, scores, classes) in enumerate(scenes, start=1):
        images.append({"id": image_id, "width": 700, "height": 700})
        for box, cls in zip(gt, gt_cls, strict=True):
            w, h = float(box[2] - box[0]), float(box[3] - box[1])
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [float(box[0]), float(box[1]), w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
        for box, score, cls in zip(boxes, scores, classes, strict=True):
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ],
                    "score": float(score),
                }
            )

    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i + 1, "name": str(i)} for i in range(4)],
        }
        truth.createIndex()
        evaluation = COCOeval(truth, truth.loadRes(detections), "bbox")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return float(evaluation.stats[1]), float(evaluation.stats[0])


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 42])
def test_evaluator_matches_pycocotools(seed: int) -> None:
    rng = np.random.default_rng(seed)
    scenes = [_scene(rng, int(rng.integers(1, 25))) for _ in range(40)]
    evaluator = DetectionEvaluator()
    for gt, gt_cls, boxes, scores, classes in scenes:
        evaluator.add(boxes, scores, classes, gt, gt_cls)

    ours = evaluator.summarize()
    map50, map50_95 = _coco_map(scenes)

    assert 0.2 < map50_95 < 0.9, "the scene should be neither trivial nor hopeless"
    assert ours["map50"] == pytest.approx(map50, abs=1e-6)
    assert ours["map50_95"] == pytest.approx(map50_95, abs=1e-6)


def test_average_precision_of_a_perfect_ranking_is_one() -> None:
    hits = np.array([True, True, False])
    assert average_precision(np.array([0.9, 0.8, 0.1], np.float32), hits, num_gt=2) == 1.0
