"""Export yolov8n to ONNX FP32/INT8, OpenVINO and TensorRT under models/."""

import argparse
import importlib.util
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
import numpy.typing as npt
import onnx

from app.models.processing import preprocess
from benchmark.coco128 import ensure_coco128, image_paths

INPUT_SIZE = 640
OPSET = 17

# The Detect head's decode tail stays FP32: its final Concat mixes 0-640 px boxes
# with 0-1 scores, and one UInt8 scale over that range rounds every score to 0.
_HEAD_PREFIX = "/model.22/"
_HEAD_CONV_BRANCHES = ("/model.22/cv2", "/model.22/cv3")


def _size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6


def _names_metadata(onnx_path: Path) -> str | None:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return next((p.value for p in model.metadata_props if p.key == "names"), None)


def _importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _should_build(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"  {path.name} exists, reusing (pass --force to rebuild)")
        return False
    return True


def export_onnx_fp32(models_dir: Path, force: bool) -> Path:
    import torch
    from ultralytics import YOLO

    onnx_path = models_dir / "yolov8n.onnx"
    if not _should_build(onnx_path, force):
        return onnx_path

    model = YOLO(str(models_dir / "yolov8n.pt"))
    exported = model.export(
        format="onnx",
        imgsz=INPUT_SIZE,
        opset=OPSET,
        dynamic=False,
        simplify=False,
        # A "cpu" string would hide the GPU from the later TensorRT build.
        device=torch.device("cpu"),
    )
    return Path(exported)


class _CalibrationReader:
    """Feeds coco128 images through the serving preprocessing to the calibrator."""

    def __init__(self, input_name: str, images: list[Path]) -> None:
        self._input_name = input_name
        self._iterator = self._batches(images)

    def _batches(self, images: list[Path]) -> Iterator[dict[str, npt.NDArray[np.float32]]]:
        import cv2

        for path in images:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            tensor, _ = preprocess(np.asarray(image, dtype=np.uint8), INPUT_SIZE)
            yield {self._input_name: tensor}

    def get_next(self) -> dict[str, npt.NDArray[np.float32]] | None:
        return next(self._iterator, None)


def export_onnx_int8(
    fp32_path: Path,
    int8_path: Path,
    calib_images: int,
    force: bool,
    keep_head_fp32: bool = True,
) -> Path:
    """Static QDQ INT8 quantisation; keep_head_fp32=False reproduces the naive collapse."""
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if not _should_build(int8_path, force):
        return int8_path

    prepared = int8_path.parent / ".cache" / "yolov8n_prepared.onnx"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    quant_pre_process(str(fp32_path), str(prepared))

    graph = onnx.load(str(prepared))
    head_tail = [
        node.name
        for node in graph.graph.node
        if keep_head_fp32
        and node.name.startswith(_HEAD_PREFIX)
        and not node.name.startswith(_HEAD_CONV_BRANCHES)
    ]
    input_name = graph.graph.input[0].name

    dataset_dir = ensure_coco128()
    images = image_paths(dataset_dir, calib_images)
    print(f"  calibrating on {len(images)} coco128 images, {len(head_tail)} head nodes kept FP32")

    quantize_static(
        model_input=str(prepared),
        model_output=str(int8_path),
        calibration_data_reader=_CalibrationReader(input_name, images),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=head_tail,
    )

    names = _names_metadata(fp32_path)
    quantized = onnx.load(str(int8_path))
    if names is not None and not any(p.key == "names" for p in quantized.metadata_props):
        quantized.metadata_props.add(key="names", value=names)
        onnx.save(quantized, str(int8_path))
    return int8_path


def export_openvino(onnx_path: Path, out_dir: Path, force: bool) -> Path:
    import openvino as ov

    from app.models.openvino_backend import NAMES_RT_INFO_KEY

    xml_path = out_dir / f"{onnx_path.stem}.xml"
    if not _should_build(xml_path, force):
        return xml_path

    model = ov.convert_model(str(onnx_path))
    names = _names_metadata(onnx_path)
    if names is not None:
        model.set_rt_info(names, NAMES_RT_INFO_KEY)
    ov.save_model(model, str(xml_path), compress_to_fp16=False)
    return xml_path


def export_tensorrt_fp16(onnx_path: Path, models_dir: Path, force: bool) -> Path:
    """Build an FP16 engine; it only runs on this GPU and TensorRT version."""
    import tensorrt as trt
    import torch

    from app.models.tensorrt_backend import engine_metadata_path

    engine_path = models_dir / "yolov8n_fp16.engine"
    if not _should_build(engine_path, force):
        return engine_path

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    cache_path = models_dir / ".cache" / "tensorrt_timing.cache"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)

    print("  building TensorRT engine (several minutes on first build) ...")
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed; see the log above.")
    build_seconds = time.perf_counter() - start

    engine_path.write_bytes(bytes(serialized))
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    engine_metadata_path(engine_path).write_text(
        json.dumps(
            {
                "precision": "fp16",
                "input_size": INPUT_SIZE,
                "names": _names_metadata(onnx_path),
                "source": onnx_path.name,
                "tensorrt_version": trt.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "build_seconds": round(build_seconds, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  built in {build_seconds:.0f}s")
    return engine_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--calib-images", type=int, default=100)
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=["openvino", "tensorrt"], help="formats to skip"
    )
    parser.add_argument("--force", action="store_true", help="rebuild existing artifacts")
    parser.add_argument(
        "--int8-naive",
        action="store_true",
        help="also write yolov8n_int8_naive.onnx with the detect head quantised too",
    )
    args = parser.parse_args()

    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    print("[1/5] ONNX FP32")
    fp32 = export_onnx_fp32(models_dir, args.force)
    artifacts["PyTorch FP32 (.pt)"] = models_dir / "yolov8n.pt"
    artifacts["ONNX FP32"] = fp32

    print("[2/5] ONNX INT8 (static, QDQ)")
    int8 = export_onnx_int8(fp32, models_dir / "yolov8n_int8.onnx", args.calib_images, args.force)
    artifacts["ONNX INT8"] = int8
    if args.int8_naive:
        naive = models_dir / "yolov8n_int8_naive.onnx"
        export_onnx_int8(fp32, naive, args.calib_images, args.force, keep_head_fp32=False)
        artifacts["ONNX INT8 (naive, head quantised)"] = naive

    if "openvino" in args.skip:
        print("[3/5] OpenVINO: skipped")
    elif _importable("openvino"):
        print("[3/5] OpenVINO FP32 + INT8")
        ov_fp32 = export_openvino(fp32, models_dir / "yolov8n_openvino", args.force)
        ov_int8 = export_openvino(int8, models_dir / "yolov8n_int8_openvino", args.force)
        artifacts["OpenVINO FP32"] = ov_fp32.parent
        artifacts["OpenVINO INT8"] = ov_int8.parent
    else:
        print("[3/5] OpenVINO: not installed (pip install '.[openvino]'), skipped")

    if "tensorrt" in args.skip:
        print("[4/5] TensorRT: skipped")
    elif _importable("tensorrt"):
        print("[4/5] TensorRT FP16")
        artifacts["TensorRT FP16"] = export_tensorrt_fp16(fp32, models_dir, args.force)
    else:
        print("[4/5] TensorRT: not installed (pip install '.[tensorrt]'), skipped")

    print("[5/5] Artifacts")
    width = max(len(name) for name in artifacts)
    for name, path in artifacts.items():
        print(f"  {name:<{width}}  {_size_mb(path):7.2f} MB  {path}")


if __name__ == "__main__":
    main()
