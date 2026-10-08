"""Settings: backend selection and the MODEL_NAME compatibility alias."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults_select_the_pytorch_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MODEL_BACKEND", "MODEL_PATH", "MODEL_NAME", "MODEL_DEVICE"):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.model_backend == "pytorch"
    assert settings.model_path == "yolov8n.pt"
    assert settings.model_device == "cpu"


def test_backend_and_path_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_BACKEND", "onnxruntime")
    monkeypatch.setenv("MODEL_PATH", "models/yolov8n_int8.onnx")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.model_backend == "onnxruntime"
    assert settings.model_path == "models/yolov8n_int8.onnx"


def test_legacy_model_name_still_sets_the_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_NAME", "yolov8s.pt")

    assert Settings(_env_file=None).model_path == "yolov8s.pt"  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("var", "value"),
    [("MODEL_BACKEND", "caffe"), ("MODEL_DEVICE", "tpu"), ("MODEL_CONFIDENCE_THRESHOLD", "1.5")],
)
def test_invalid_values_are_rejected_at_startup(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    monkeypatch.setenv(var, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
