"""When the export script reuses an artifact and when it rebuilds one."""

import os
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from scripts.export_models import _has_fixed_batch, _should_build


def _identity_model(path: Path, batch: int | str) -> Path:
    shape = [batch, 3, 640, 640]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, shape)],
    )
    onnx.save(helper.make_model(graph), str(path))
    return path


def test_a_fixed_batch_export_is_detected(tmp_path: Path) -> None:
    assert _has_fixed_batch(_identity_model(tmp_path / "static.onnx", 1))
    assert not _has_fixed_batch(_identity_model(tmp_path / "dynamic.onnx", "batch"))


def test_missing_or_forced_artifacts_are_built(tmp_path: Path) -> None:
    artifact = tmp_path / "model.engine"
    assert _should_build(artifact, force=False)
    artifact.write_bytes(b"x")
    assert _should_build(artifact, force=True)


def test_an_artifact_newer_than_its_source_is_reused(tmp_path: Path) -> None:
    source, artifact = tmp_path / "model.onnx", tmp_path / "model.engine"
    source.write_bytes(b"x")
    artifact.write_bytes(b"x")
    os.utime(source, (1_000, 1_000))
    os.utime(artifact, (2_000, 2_000))

    assert not _should_build(artifact, force=False, source=source)


def test_an_artifact_older_than_its_source_is_rebuilt(tmp_path: Path) -> None:
    source, artifact = tmp_path / "model.onnx", tmp_path / "model.engine"
    source.write_bytes(b"x")
    artifact.write_bytes(b"x")
    os.utime(source, (2_000, 2_000))
    os.utime(artifact, (1_000, 1_000))

    assert _should_build(artifact, force=False, source=source)
