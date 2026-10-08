"""Benchmark every available backend: accuracy, latency, memory and HTTP.

    python -m benchmark.run_benchmark            # full run, ~10 min
    python -m benchmark.run_benchmark --quick    # smoke run, ~2 min

Run ``python -m scripts.export_models`` first. Backends whose artifact or
package is missing are reported as skipped, never estimated.

Each backend's model-level measurements run in a fresh subprocess (the
``--worker`` mode below), so peak memory belongs to that backend alone and
no CUDA or thread-pool state leaks from one backend into the next.

Results go to ``benchmark/results/<hardware-tag>[-quick].{json,md}``.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmark import hardware
from benchmark.coco128 import ensure_coco128, image_paths, iter_samples
from benchmark.metrics import DetectionEvaluator

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Serving thresholds: what the API runs with, so latency is measured here.
SERVE_CONF, SERVE_IOU, SERVE_MAX_DET = 0.25, 0.7, 300
# Evaluation thresholds: the standard low confidence floor for mAP, so the
# precision-recall curve is traced all the way down. Identical for every
# backend.
EVAL_CONF, EVAL_IOU, EVAL_MAX_DET = 0.001, 0.7, 300


@dataclass(frozen=True)
class Config:
    """One row of the benchmark: a backend, a device and an artifact."""

    name: str
    backend: str
    device: str
    model_path: str
    package: str
    # Ablations are scored for accuracy only; they are not serving candidates.
    accuracy_only: bool = False


CONFIGS = (
    Config("pytorch-cpu-fp32", "pytorch", "cpu", "models/yolov8n.pt", "torch"),
    Config("pytorch-cuda-fp32", "pytorch", "cuda", "models/yolov8n.pt", "torch"),
    Config("onnxruntime-cpu-fp32", "onnxruntime", "cpu", "models/yolov8n.onnx", "onnxruntime"),
    Config("onnxruntime-cpu-int8", "onnxruntime", "cpu", "models/yolov8n_int8.onnx", "onnxruntime"),
    Config("onnxruntime-cuda-fp32", "onnxruntime", "cuda", "models/yolov8n.onnx", "onnxruntime"),
    Config("openvino-cpu-fp32", "openvino", "cpu", "models/yolov8n_openvino/yolov8n.xml", "openvino"),
    Config("openvino-cpu-int8", "openvino", "cpu", "models/yolov8n_int8_openvino/yolov8n_int8.xml", "openvino"),
    Config("tensorrt-cuda-fp16", "tensorrt", "cuda", "models/yolov8n_fp16.engine", "tensorrt"),
    Config("onnxruntime-cpu-int8-naive", "onnxruntime", "cpu", "models/yolov8n_int8_naive.onnx", "onnxruntime", accuracy_only=True),
)  # fmt: skip


@dataclass(frozen=True)
class Budget:
    """How much work each phase does; ``--quick`` shrinks every number."""

    eval_images: int | None  # None = all 128
    warmup_runs: int
    timed_runs: int
    http_requests: int
    http_warmup: int
    http_concurrency: tuple[int, ...]


FULL = Budget(None, 10, 100, 200, 5, (1, 8))
QUICK = Budget(16, 3, 20, 20, 2, (1, 8))


# ── Worker (runs inside a fresh subprocess) ──────────────────


def _peak_rss_mb() -> float:
    """Peak resident memory of this process so far."""
    if sys.platform == "win32":
        import psutil

        return float(psutil.Process().memory_info().peak_wset) / 1e6

    import resource

    # ru_maxrss is KiB on Linux, bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * (1 if sys.platform == "darwin" else 1024) / 1e6


class _GpuMemorySampler:
    """Polls device memory in the background and records the peak.

    NVML reports memory in use on the whole device, so this is measured as
    peak minus the level before the model loaded. Other processes changing
    their usage during the run would show up here too.
    """

    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.baseline = self._used()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _used(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def _run(self) -> None:
        while not self._stop.wait(0.01):
            self.peak = max(self.peak, self._used())

    def stop(self) -> float:
        """Stop sampling; return the peak increase in MB."""
        self._stop.set()
        self._thread.join()
        self._nvml.nvmlShutdown()
        return (self.peak - self.baseline) / 1e6


def run_worker(config: Config, budget: Budget) -> dict[str, Any]:
    """Load one backend and measure latency, accuracy and memory."""
    from app.core.logging import setup_logging
    from app.models.factory import create_backend

    setup_logging(log_level="WARNING", log_format="json")
    sampler = _GpuMemorySampler() if config.device == "cuda" else None

    backend = create_backend(
        config.backend,
        config.model_path,
        config.device,
        confidence_threshold=SERVE_CONF,
        iou_threshold=SERVE_IOU,
        max_detections=SERVE_MAX_DET,
    )
    start = time.perf_counter()
    backend.load()
    result: dict[str, Any] = {
        "precision": backend.precision,
        "load_s": round(time.perf_counter() - start, 2),
    }

    dataset_dir = ensure_coco128()

    # ── Latency: serving thresholds, one fixed image, batch 1 ───
    if not config.accuracy_only:
        import cv2

        image_path = image_paths(dataset_dir, 1)[0]
        image = np.asarray(cv2.imread(str(image_path), cv2.IMREAD_COLOR), dtype=np.uint8)
        for _ in range(budget.warmup_runs):
            backend.predict(image)
        timings = []
        for _ in range(budget.timed_runs):
            t0 = time.perf_counter()
            backend.predict(image)
            timings.append((time.perf_counter() - t0) * 1000)
        ms = np.asarray(timings)
        result["latency"] = {
            "image": image_path.name,
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "warmup_runs": budget.warmup_runs,
            "timed_runs": budget.timed_runs,
            "mean_ms": round(float(ms.mean()), 2),
            "p50_ms": round(float(np.percentile(ms, 50)), 2),
            "p95_ms": round(float(np.percentile(ms, 95)), 2),
            "throughput_img_s": round(1000.0 / float(ms.mean()), 1),
        }

    # ── Accuracy: evaluation thresholds, same code for all ──────
    backend.confidence_threshold = EVAL_CONF
    backend.iou_threshold = EVAL_IOU
    backend.max_detections = EVAL_MAX_DET
    evaluator = DetectionEvaluator()
    for sample in iter_samples(dataset_dir, budget.eval_images):
        detections = backend.predict(sample.image)
        evaluator.add(
            np.asarray(
                [[d.x_min, d.y_min, d.x_max, d.y_max] for d in detections], dtype=np.float32
            ).reshape(-1, 4),
            np.asarray([d.confidence for d in detections], dtype=np.float32),
            np.asarray([d.class_id for d in detections], dtype=np.int64),
            sample.boxes,
            sample.classes,
        )
    accuracy = evaluator.summarize()
    result["accuracy"] = {
        "map50": round(accuracy["map50"], 4),
        "map50_95": round(accuracy["map50_95"], 4),
        "images": accuracy["num_images"],
        "gt_boxes": accuracy["num_gt_boxes"],
    }

    result["peak_rss_mb"] = round(_peak_rss_mb(), 1)
    if sampler is not None:
        result["peak_gpu_mem_mb"] = round(sampler.stop(), 1)
    backend.unload()
    return result


# ── Orchestrator ─────────────────────────────────────────────


def _artifact_size_mb(path: Path) -> float:
    # An OpenVINO IR is an .xml graph plus a .bin of weights.
    files = [path, path.with_suffix(".bin")] if path.suffix == ".xml" else [path]
    return round(sum(f.stat().st_size for f in files if f.exists()) / 1e6, 2)


def _unavailable_reason(config: Config, gpu: dict[str, Any] | None) -> str | None:
    if importlib.util.find_spec(config.package) is None:
        return f"package '{config.package}' not installed"
    if not Path(config.model_path).exists():
        return f"{config.model_path} missing (run python -m scripts.export_models)"
    if config.device == "cuda" and gpu is None:
        return "no NVIDIA GPU detected"
    return None


def _run_in_subprocess(config: Config, quick: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = [sys.executable, "-m", "benchmark.run_benchmark", "--worker", config.name]
        cmd += ["--result-file", str(out)] + (["--quick"] if quick else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0 or not out.exists():
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-8:])
            raise RuntimeError(f"worker failed (exit {proc.returncode}):\n{tail}")
        result: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        return result


def run_all(quick: bool, only: list[str] | None, skip_http: bool) -> dict[str, Any]:
    from benchmark.http_load import run_http_benchmark
    from benchmark.report import render_markdown

    budget = QUICK if quick else FULL
    info = hardware.collect()
    tag = hardware.hardware_tag(info) + ("-quick" if quick else "")
    dataset_dir = ensure_coco128()
    http_image = image_paths(dataset_dir, 1)[0].read_bytes()

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        if only and config.name not in only:
            continue
        row: dict[str, Any] = {**asdict(config)}
        reason = _unavailable_reason(config, info["gpu"])
        if reason:
            print(f"- {config.name}: skipped ({reason})")
            rows.append({**row, "status": "skipped", "reason": reason})
            continue

        print(f"- {config.name}: model ...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            row.update(_run_in_subprocess(config, quick))
            row["size_mb"] = _artifact_size_mb(Path(config.model_path))
            if not (config.accuracy_only or skip_http):
                print(" http ...", end="", flush=True)
                http = run_http_benchmark(
                    config.backend,
                    config.model_path,
                    config.device,
                    http_image,
                    budget.http_requests,
                    budget.http_concurrency,
                    budget.http_warmup,
                )
                # The server must be running what this row claims it is.
                served = (http["health"]["backend"], http["health"]["precision"])
                if served != (config.backend, row["precision"]):
                    raise RuntimeError(f"server reported {served}, expected {config.backend}")
                row["http"] = http["runs"]
            row["status"] = "ok"
        except Exception as exc:
            row.update(status="failed", reason=str(exc))
        print(f" {row['status']} ({time.perf_counter() - t0:.0f}s)")
        if row["status"] == "failed":
            print(f"    {row['reason']}")
        rows.append(row)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "quick" if quick else "full",
        "duration_s": round(time.perf_counter() - started, 1),
        "hardware": info,
        "settings": {
            "budget": asdict(budget),
            "input_size": 640,
            "serving_thresholds": {"conf": SERVE_CONF, "iou": SERVE_IOU, "max_det": SERVE_MAX_DET},
            "eval_thresholds": {"conf": EVAL_CONF, "iou": EVAL_IOU, "max_det": EVAL_MAX_DET},
            "dataset": "coco128 (first 128 images of COCO train2017)",
        },
        "results": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"{tag}.json", RESULTS_DIR / f"{tag}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nDone in {report['duration_s']:.0f}s -> {json_path}, {md_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="smaller budget, ~2 minutes")
    parser.add_argument("--only", nargs="*", help="run only these config names")
    parser.add_argument("--skip-http", action="store_true", help="skip the HTTP load test")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        config = next(c for c in CONFIGS if c.name == args.worker)
        result = run_worker(config, QUICK if args.quick else FULL)
        args.result_file.write_text(json.dumps(result), encoding="utf-8")
        return

    run_all(args.quick, args.only, args.skip_http)


if __name__ == "__main__":
    main()
