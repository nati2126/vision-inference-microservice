"""Every backend: same output schema, correct precision, and parity with a reference.

The reference is ONNX Runtime FP32 on CPU, i.e. the same network weights run
through the shared NumPy pre/post-processing. Each other backend must find
the same confident objects within a tolerance that fits its precision.

Backends whose package, artifact or GPU is unavailable are skipped; CI runs
the PyTorch and ONNX Runtime FP32 CPU cases.
"""

import importlib.util
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from app.models.base import Detection, InferenceBackend
from app.models.factory import create_backend
from app.models.labels import COCO_CLASSES
from app.schemas.detection import Detection as DetectionSchema
from benchmark.metrics import box_iou
from tests.conftest import MODELS_DIR


@dataclass(frozen=True)
class Case:
    """A backend under test and how closely it must match the reference."""

    id: str
    backend: str
    device: str
    artifact: str  # path relative to models/
    precision: str
    # Tolerances, set at roughly twice the worst case measured on bus.jpg
    # and zidane.jpg (see the PR / README for the measurements).
    min_iou: float
    max_conf_diff: float
    package: str


CASES = [
    # PyTorch letterboxes to a minimal rectangle (e.g. 640x480) instead of
    # the exported models' fixed 640x640, so its inputs differ slightly.
    # Measured worst case: IoU 0.950, confidence 0.058.
    Case("pytorch-cpu", "pytorch", "cpu", "yolov8n.pt", "fp32", 0.90, 0.10, "torch"),
    # Same graph, different kernels: measured IoU >= 0.9996, conf <= 0.0005.
    Case("onnxruntime-cpu-fp32", "onnxruntime", "cpu", "yolov8n.onnx", "fp32", 0.99, 0.01, "onnxruntime"),
    Case("onnxruntime-cuda-fp32", "onnxruntime", "cuda", "yolov8n.onnx", "fp32", 0.99, 0.01, "onnxruntime"),
    Case("openvino-cpu-fp32", "openvino", "cpu", "yolov8n_openvino/yolov8n.xml", "fp32", 0.99, 0.01, "openvino"),
    # FP16: measured IoU >= 0.999, conf <= 0.0018.
    Case("tensorrt-cuda-fp16", "tensorrt", "cuda", "yolov8n_fp16.engine", "fp16", 0.98, 0.02, "tensorrt"),
    # INT8: measured IoU >= 0.974, conf <= 0.049.
    Case("onnxruntime-cpu-int8", "onnxruntime", "cpu", "yolov8n_int8.onnx", "int8", 0.90, 0.10, "onnxruntime"),
    Case("openvino-cpu-int8", "openvino", "cpu", "yolov8n_int8_openvino/yolov8n_int8.xml", "int8", 0.90, 0.10, "openvino"),
]  # fmt: skip


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.fixture(scope="session", params=CASES, ids=[c.id for c in CASES])
def case_and_backend(
    request: pytest.FixtureRequest, onnx_fp32_path: Path, pt_path: Path
) -> Iterator[tuple[Case, InferenceBackend]]:
    """Each available backend, loaded once for the whole session."""
    case: Case = request.param
    if importlib.util.find_spec(case.package) is None:
        pytest.skip(f"{case.package} is not installed")
    if case.device == "cuda" and not _cuda_available():
        pytest.skip("no CUDA device")
    artifact = MODELS_DIR / case.artifact
    if not artifact.exists():
        pytest.skip(f"{artifact} not exported (python -m scripts.export_models)")

    backend = create_backend(case.backend, str(artifact), case.device)
    backend.load()
    yield case, backend
    backend.unload()


@pytest.fixture(scope="session")
def reference(onnx_fp32_path: Path) -> InferenceBackend:
    backend = create_backend("onnxruntime", str(onnx_fp32_path), "cpu")
    backend.load()
    return backend


# ── Schema ───────────────────────────────────────────────────


def test_backend_returns_the_shared_detection_schema(
    case_and_backend: tuple[Case, InferenceBackend], bus_image: npt.NDArray[np.uint8]
) -> None:
    _, backend = case_and_backend
    height, width = bus_image.shape[:2]

    detections = backend.predict(bus_image)

    assert detections, "the bus photo should produce detections"
    for det in detections:
        assert isinstance(det, Detection)
        assert det.label in COCO_CLASSES
        assert COCO_CLASSES[det.class_id] == det.label
        assert backend.confidence_threshold <= det.confidence <= 1.0
        assert 0 <= det.x_min < det.x_max <= width
        assert 0 <= det.y_min < det.y_max <= height
        # Serialises into the API's response model without error.
        DetectionSchema(**det.to_dict())

    labels = [d.label for d in detections]
    assert "bus" in labels
    assert labels.count("person") >= 3


def test_backend_reports_precision_from_the_artifact(
    case_and_backend: tuple[Case, InferenceBackend],
) -> None:
    case, backend = case_and_backend

    assert backend.precision == case.precision
    assert backend.name == case.backend


# ── Parity ───────────────────────────────────────────────────


def _assert_confident_detections_match(
    expected: list[Detection], actual: list[Detection], min_iou: float, max_conf_diff: float
) -> None:
    """Every detection >= 0.5 in ``expected`` has a same-class twin in ``actual``.

    Candidates are drawn from everything ``actual`` returned (>= 0.25), so a
    box sitting just below 0.5 on one side does not fail spuriously.
    """
    for want in (d for d in expected if d.confidence >= 0.5):
        same_class = [d for d in actual if d.class_id == want.class_id]
        assert same_class, f"no {want.label} found to match {want}"
        ious = box_iou(
            np.asarray([[want.x_min, want.y_min, want.x_max, want.y_max]], dtype=np.float32),
            np.asarray([[d.x_min, d.y_min, d.x_max, d.y_max] for d in same_class], np.float32),
        )[0]
        best = int(ious.argmax())
        assert ious[best] >= min_iou, f"{want.label}: IoU {ious[best]:.4f} < {min_iou}"
        conf_diff = abs(same_class[best].confidence - want.confidence)
        assert conf_diff <= max_conf_diff, f"{want.label}: confidence differs by {conf_diff:.4f}"


@pytest.mark.parametrize("image_name", ["bus_image", "zidane_image"])
def test_backend_matches_onnx_fp32_reference(
    case_and_backend: tuple[Case, InferenceBackend],
    reference: InferenceBackend,
    image_name: str,
    request: pytest.FixtureRequest,
) -> None:
    case, backend = case_and_backend
    image = request.getfixturevalue(image_name)

    want, got = reference.predict(image), backend.predict(image)

    # Both directions: nothing confident is missing, nothing confident is extra.
    _assert_confident_detections_match(want, got, case.min_iou, case.max_conf_diff)
    _assert_confident_detections_match(got, want, case.min_iou, case.max_conf_diff)


def test_numpy_pipeline_matches_ultralytics_on_the_same_onnx_file(
    reference: InferenceBackend, onnx_fp32_path: Path, bus_image: npt.NDArray[np.uint8]
) -> None:
    """Our letterbox/decode/NMS reproduces ultralytics' own ONNX path.

    Same file, same runtime, so any difference would be a pre/post-processing
    bug; the tolerance only absorbs float rounding.
    """
    import torch
    from ultralytics import YOLO

    result = YOLO(str(onnx_fp32_path), task="detect").predict(
        bus_image, device=torch.device("cpu"), verbose=False
    )[0]
    assert result.boxes is not None
    theirs = sorted(
        zip(
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.xyxy.tolist(),
            strict=False,
        ),
        reverse=True,
    )
    ours = sorted(
        (
            (d.confidence, d.class_id, [d.x_min, d.y_min, d.x_max, d.y_max])
            for d in reference.predict(bus_image)
        ),
        reverse=True,
    )

    assert len(ours) == len(theirs)
    for (conf_a, cls_a, box_a), (conf_b, cls_b, box_b) in zip(ours, theirs, strict=True):
        assert cls_a == int(cls_b)
        assert conf_a == pytest.approx(conf_b, abs=1e-4)
        np.testing.assert_allclose(box_a, box_b, atol=0.05)


# ── Construction errors ──────────────────────────────────────


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("caffe", "model.caffemodel")


def test_onnxruntime_rejects_unsupported_device() -> None:
    with pytest.raises(ValueError, match="supports"):
        create_backend("onnxruntime", "model.onnx", device="mps")


@pytest.mark.parametrize("backend", ["onnxruntime", "openvino"])
def test_predict_before_load_raises(backend: str) -> None:
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"{backend} is not installed")
    with pytest.raises(RuntimeError, match="not loaded"):
        create_backend(backend, "unused.onnx").predict(np.zeros((8, 8, 3), np.uint8))
