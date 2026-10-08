"""Unit tests for the PyTorch backend that do not need real weights."""

from pathlib import Path

import numpy as np
import pytest

from app.models.pytorch_backend import PyTorchBackend


def test_predict_before_load_raises() -> None:
    """Calling predict() on an unloaded model is a RuntimeError, not a crash."""
    model = PyTorchBackend()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="not loaded"):
        model.predict(image)


def test_is_loaded_false_before_load() -> None:
    assert PyTorchBackend().is_loaded is False


def test_bare_filename_resolves_into_weights_dir(tmp_path: Path) -> None:
    """A bare filename becomes an absolute path under weights_dir.

    This is what keeps ultralytics from downloading into the working
    directory, which is not writable by the container's non-root user.
    """
    model = PyTorchBackend(model_path="yolov8n.pt", weights_dir=tmp_path / "weights")

    resolved = Path(model._resolve_weights())

    assert resolved.is_absolute()
    assert resolved == tmp_path / "weights" / "yolov8n.pt"
    assert resolved.parent.is_dir(), "weights_dir should be created eagerly"


def test_explicit_path_is_passed_through(tmp_path: Path) -> None:
    """An explicit path in model_path is respected, not rewritten."""
    explicit = tmp_path / "custom" / "best.pt"
    model = PyTorchBackend(model_path=str(explicit), weights_dir=tmp_path / "weights")

    assert model._resolve_weights() == str(explicit)


def test_without_weights_dir_name_is_unchanged() -> None:
    """With no weights_dir configured the name reaches ultralytics as-is."""
    model = PyTorchBackend(model_path="yolov8n.pt", weights_dir=None)

    assert model._resolve_weights() == "yolov8n.pt"
