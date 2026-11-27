"""Accuracy on COCO val2017 (5000 images), scored by pycocotools.

Every backend runs the service's own pipeline (``predict()``) at the
evaluation thresholds, in its own process. The detections are scored twice:
by pycocotools, the reference implementation, and by this repository's
``DetectionEvaluator``, which scores the coco128 numbers in the main
benchmark. The second score checks that evaluator against the reference.

    pip install -e ".[bench]"
    python -m benchmark.coco_val                      # ~1 GB download on first run
    python -m benchmark.coco_val --only tensorrt-cuda-fp16 --limit 500
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmark import hardware
from benchmark.coco128 import DEFAULT_ROOT
from benchmark.metrics import DetectionEvaluator
from benchmark.run_benchmark import (
    CONFIGS,
    EVAL_CONF,
    EVAL_IOU,
    EVAL_MAX_DET,
    RESULTS_DIR,
    Config,
    _unavailable_reason,
)

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"


def _download(url: str, target: Path) -> None:
    print(f"Downloading {url} ...", flush=True)
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.rename(target)


def _extract(archive_path: Path, root: Path, members: list[str] | None = None) -> None:
    resolved_root = root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = members or archive.namelist()
        for name in names:
            if not (root / name).resolve().is_relative_to(resolved_root):
                raise RuntimeError(f"Unsafe path in {archive_path.name}: {name!r}")
        archive.extractall(root, names)


def ensure_val2017(root: Path = DEFAULT_ROOT / "coco") -> tuple[Path, Path]:
    """Download and extract val2017 if needed; return (images dir, annotations file)."""
    images, annotations = root / "val2017", root / ANNOTATIONS_MEMBER
    root.mkdir(parents=True, exist_ok=True)
    if not annotations.exists():
        archive = root / "annotations_trainval2017.zip"
        if not archive.exists():
            _download(ANNOTATIONS_URL, archive)
        _extract(archive, root, [ANNOTATIONS_MEMBER])
    if not images.is_dir() or len(list(images.glob("*.jpg"))) < 5000:
        archive = root / "val2017.zip"
        if not archive.exists():
            _download(IMAGES_URL, archive)
        _extract(archive, root)
    return images, annotations


def _class_to_category(annotations: dict[str, Any]) -> list[int]:
    """YOLO class index -> COCO category id; YOLO orders the 80 classes by id."""
    from app.models.labels import COCO_CLASSES

    categories = sorted(annotations["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in categories]
    if names != list(COCO_CLASSES):
        raise RuntimeError("COCO category names do not match the model's class list")
    return [c["id"] for c in categories]


def run_worker(config: Config, limit: int | None) -> dict[str, Any]:
    """Predict on val2017; return COCO-format detections and the in-house scores."""
    import cv2

    from app.core.logging import setup_logging
    from app.models.factory import create_backend

    setup_logging(log_level="WARNING", log_format="json")
    images_dir, annotations_path = ensure_val2017()
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    category_ids = _class_to_category(annotations)
    class_of = {cat: cls for cls, cat in enumerate(category_ids)}

    # Crowd regions are ignored by pycocotools; the in-house evaluator has no
    # notion of them, so it gets the non-crowd boxes only.
    gt: dict[int, list[tuple[list[float], int]]] = {}
    for ann in annotations["annotations"]:
        if not ann["iscrowd"]:
            x, y, w, h = ann["bbox"]
            gt.setdefault(ann["image_id"], []).append(
                ([x, y, x + w, y + h], class_of[ann["category_id"]])
            )
    images = sorted(annotations["images"], key=lambda i: i["id"])[:limit]

    backend = create_backend(
        config.backend,
        config.model_path,
        config.device,
        confidence_threshold=EVAL_CONF,
        iou_threshold=EVAL_IOU,
        max_detections=EVAL_MAX_DET,
    )
    backend.load()

    detections: list[dict[str, Any]] = []
    evaluator = DetectionEvaluator()
    start = time.perf_counter()
    for info in images:
        image = cv2.imread(str(images_dir / info["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {info['file_name']}")
        found = backend.predict(np.asarray(image, dtype=np.uint8))
        for d in found:
            detections.append(
                {
                    "image_id": info["id"],
                    "category_id": category_ids[d.class_id],
                    "bbox": [
                        round(d.x_min, 2),
                        round(d.y_min, 2),
                        round(d.x_max - d.x_min, 2),
                        round(d.y_max - d.y_min, 2),
                    ],
                    "score": round(d.confidence, 5),
                }
            )
        truth = gt.get(info["id"], [])
        evaluator.add(
            np.asarray([[d.x_min, d.y_min, d.x_max, d.y_max] for d in found], np.float32).reshape(
                -1, 4
            ),
            np.asarray([d.confidence for d in found], np.float32),
            np.asarray([d.class_id for d in found], np.int64),
            np.asarray([box for box, _ in truth], np.float32).reshape(-1, 4),
            np.asarray([cls for _, cls in truth], np.int64),
        )
    seconds = time.perf_counter() - start
    in_house = evaluator.summarize()
    return {
        "precision": backend.precision,
        "image_ids": [info["id"] for info in images],
        "detections": detections,
        "seconds": round(seconds, 1),
        "in_house": {"map50": in_house["map50"], "map50_95": in_house["map50_95"]},
    }


def score(
    annotations_path: Path, image_ids: list[int], detections: list[dict[str, Any]]
) -> dict[str, float]:
    """pycocotools' bbox evaluation over the given images."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(annotations_path))
        predicted = truth.loadRes(detections) if detections else COCO()
        evaluation = COCOeval(truth, predicted, "bbox")
        evaluation.params.imgIds = image_ids
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    names = ["map50_95", "map50", "map75", "map_small", "map_medium", "map_large"]
    return {name: float(value) for name, value in zip(names, evaluation.stats[:6], strict=True)}


def _run_in_subprocess(config: Config, limit: int | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = [sys.executable, "-m", "benchmark.coco_val", "--worker", config.name]
        cmd += ["--result-file", str(out)] + (["--limit", str(limit)] if limit else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0 or not out.exists():
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-8:])
            raise RuntimeError(f"worker failed (exit {proc.returncode}):\n{tail}")
        result: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        return result


def render_markdown(report: dict[str, Any]) -> str:
    from benchmark.report import hardware_line

    lines = [
        "# COCO val2017 accuracy",
        "",
        f"- **Hardware:** {hardware_line(report)}",
        f"- **Generated:** {report['generated_at']}, commit `{report['hardware'].get('git_commit', '?')}`,"
        f" took {report['duration_s']:.0f}s",
        f"- **Data:** COCO val2017, {report['settings']['images']} images,"
        f" conf {EVAL_CONF}, NMS IoU {EVAL_IOU}, max {EVAL_MAX_DET} detections, 640x640 letterbox.",
        "- **Scoring:** pycocotools (`COCOeval`, bbox). The in-house column is this repository's"
        " `DetectionEvaluator` on the same detections (non-crowd ground truth).",
        "",
        "| Backend | Device | Precision | mAP50-95 | mAP50 | mAP75 | small | medium | large"
        " | In-house mAP50-95 | In-house mAP50 | Images/s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        if row["status"] != "ok":
            lines.append(
                f"| {row['backend']} | {row['device']} | {row['status']}: {row['reason']} |"
            )
            continue
        c, h = row["coco"], row["in_house"]
        name = row["backend"] + (" (ablation: naive INT8)" if row["accuracy_only"] else "")
        lines.append(
            f"| {name} | {row['device']} | {row['precision']} | {c['map50_95']:.3f} | {c['map50']:.3f}"
            f" | {c['map75']:.3f} | {c['map_small']:.3f} | {c['map_medium']:.3f} | {c['map_large']:.3f}"
            f" | {h['map50_95']:.3f} | {h['map50']:.3f} | {row['images_per_s']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def run_all(only: list[str] | None, limit: int | None) -> None:
    info = hardware.collect()
    _, annotations_path = ensure_val2017()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    images = 0
    for config in CONFIGS:
        if only and config.name not in only:
            continue
        row: dict[str, Any] = {
            "name": config.name,
            "backend": config.backend,
            "device": config.device,
            "accuracy_only": config.accuracy_only,
        }
        reason = _unavailable_reason(config, info["gpu"])
        if reason:
            print(f"- {config.name}: skipped ({reason})")
            rows.append({**row, "status": "skipped", "reason": reason})
            continue
        print(f"- {config.name} ...", end="", flush=True)
        try:
            result = _run_in_subprocess(config, limit)
            images = len(result["image_ids"])
            row.update(
                status="ok",
                precision=result["precision"],
                coco=score(annotations_path, result["image_ids"], result["detections"]),
                in_house=result["in_house"],
                images_per_s=round(images / result["seconds"], 1),
            )
            print(f" mAP50-95 {row['coco']['map50_95']:.4f} ({result['seconds']:.0f}s)")
        except Exception as exc:
            row.update(status="failed", reason=str(exc))
            print(f" failed\n    {exc}")
        rows.append(row)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - started, 1),
        "hardware": info,
        "settings": {
            "dataset": "COCO val2017",
            "images": images,
            "eval_thresholds": {"conf": EVAL_CONF, "iou": EVAL_IOU, "max_det": EVAL_MAX_DET},
        },
        "results": rows,
    }
    tag = hardware.hardware_tag(info) + "-coco-val2017" + (f"-{limit}" if limit else "")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"{tag}.json", RESULTS_DIR / f"{tag}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nDone in {report['duration_s']:.0f}s -> {json_path}, {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="run only these config names")
    parser.add_argument("--limit", type=int, help="first N images by id, for a quick check")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        config = next(c for c in CONFIGS if c.name == args.worker)
        args.result_file.write_text(json.dumps(run_worker(config, args.limit)), encoding="utf-8")
        return
    run_all(args.only, args.limit)


if __name__ == "__main__":
    main()
