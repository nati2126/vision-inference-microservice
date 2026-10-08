"""Export yolov8n to every backend format used by the service and benchmark.

    python -m scripts.export_models [--skip openvino tensorrt] [--force]

Produces, under ``models/``:

    yolov8n.pt                      PyTorch weights (downloaded)
    yolov8n.onnx                    ONNX FP32, static 1x3x640x640, opset 17
    yolov8n_int8.onnx               ONNX INT8, static QDQ quantisation
    yolov8n_openvino/               OpenVINO IR FP32          [openvino extra]
    yolov8n_int8_openvino/          OpenVINO IR from the INT8 ONNX
    yolov8n_fp16.engine (+ .json)   TensorRT FP16 engine      [tensorrt extra]

Existing artifacts are reused unless ``--force`` is given; the TensorRT
build in particular takes minutes.
"""

import argparse
import importlib.util
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

# Ultralytics pip-installs missing packages on the fly by default. An export
# script must fail loudly instead of mutating the environment.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
import numpy.typing as npt
import onnx

from app.models.processing import preprocess
from benchmark.coco128 import ensure_coco128, image_paths

INPUT_SIZE = 640
OPSET = 17

# The decode tail of YOLOv8's Detect head (model.22), kept in float during
# INT8 quantisation. Its last Concat joins box coordinates (0..640 px) with
# sigmoid class scores (0..1) in a single tensor; one per-tensor UInt8 scale
# covering 0..640 has a step of ~2.5, which rounds every class score to 0 or
# 2.5 and destroys accuracy. The 64 convolutions, including the head's own
# cv2/cv3 branches, are still quantised.
_HEAD_PREFIX = "/model.22/"
_HEAD_CONV_BRANCHES = ("/model.22/cv2", "/model.22/cv3")


# ── Helpers ──────────────────────────────────────────────────


def _size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6


def _names_metadata(onnx_path: Path) -> str | None:
    """The class-name metadata ultralytics writes into its ONNX export."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    return next((p.value for p in model.metadata_props if p.key == "names"), None)


def _importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _should_build(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"  {path.name} exists, reusing (pass --force to rebuild)")
        return False
    return True


# ── PyTorch / ONNX FP32 ──────────────────────────────────────


def export_onnx_fp32(models_dir: Path, force: bool) -> Path:
    """Download yolov8n.pt and export it to a static-shape ONNX FP32 graph."""
    import torch
    from ultralytics import YOLO

    onnx_path = models_dir / "yolov8n.onnx"
    if not _should_build(onnx_path, force):
        return onnx_path

    # Ultralytics downloads a known asset name to the path given.
    model = YOLO(str(models_dir / "yolov8n.pt"))
    exported = model.export(
        format="onnx",
        imgsz=INPUT_SIZE,
        opset=OPSET,
        dynamic=False,  # static shapes: required by TensorRT, faster everywhere
        simplify=False,  # onnxslim is not a dependency; ORT optimises at load
        # A torch.device, not "cpu": the string makes ultralytics set
        # CUDA_VISIBLE_DEVICES=-1 for the whole process, and the TensorRT
        # build later in this script would then find no GPU.
        device=torch.device("cpu"),
    )
    return Path(exported)


# ── ONNX INT8 (static quantisation) ──────────────────────────


class _CalibrationReader:
    """Feeds preprocessed coco128 images to the ORT calibrator, one at a time.

    The images go through exactly the same letterbox/normalisation as at
    inference time, so the observed activation ranges match what the
    quantised model will actually see.
    """

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
    """Static INT8 quantisation of the FP32 graph with ONNX Runtime.

    *Static* means activation ranges are measured once, offline, on
    calibration data, and baked into the graph as fixed scales and zero
    points. (Dynamic quantisation computes them per inference, which costs
    time and is aimed at MatMul-heavy models, not CNNs.)

    Format: QDQ, i.e. QuantizeLinear/DequantizeLinear pairs around each
    quantised op. Runtimes fuse the pairs into INT8 kernels, and the same
    file also runs on OpenVINO.

    ``keep_head_fp32=False`` quantises the decode tail too; it exists only
    to reproduce the accuracy collapse described at ``_HEAD_PREFIX``.
    """
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
    # Shape inference + graph optimisation (e.g. folding BatchNorm) before
    # quantising, as ORT recommends; otherwise the quantiser sees ops that
    # would have been fused away and calibrates the wrong tensors.
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
        # U8 activations / S8 weights: the combination x86 VNNI kernels are
        # built for. Per-channel weight scales track each filter's own range,
        # which matters for depthwise-heavy and small models like yolov8n.
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=head_tail,
    )

    # Carry the class names over; the quantiser does not copy metadata.
    names = _names_metadata(fp32_path)
    quantized = onnx.load(str(int8_path))
    if names is not None and not any(p.key == "names" for p in quantized.metadata_props):
        quantized.metadata_props.add(key="names", value=names)
        onnx.save(quantized, str(int8_path))
    return int8_path


# ── OpenVINO ─────────────────────────────────────────────────


def export_openvino(onnx_path: Path, out_dir: Path, force: bool) -> Path:
    """Convert an ONNX file (FP32 or QDQ INT8) to OpenVINO IR."""
    import openvino as ov

    from app.models.openvino_backend import NAMES_RT_INFO_KEY

    xml_path = out_dir / f"{onnx_path.stem}.xml"
    if not _should_build(xml_path, force):
        return xml_path

    model = ov.convert_model(str(onnx_path))
    names = _names_metadata(onnx_path)
    if names is not None:
        model.set_rt_info(names, NAMES_RT_INFO_KEY)
    # Keep FP32 weights: compress_to_fp16 defaults to True and would make
    # the "FP32" row of the benchmark something else.
    ov.save_model(model, str(xml_path), compress_to_fp16=False)
    return xml_path


# ── TensorRT ─────────────────────────────────────────────────


def export_tensorrt_fp16(onnx_path: Path, models_dir: Path, force: bool) -> Path:
    """Build a TensorRT FP16 engine from the FP32 ONNX graph.

    FP16 is not a separate model: TensorRT is *allowed* to pick FP16 kernels
    for each layer and keeps FP32 where that is faster or required. Engines
    are tied to this GPU and TensorRT version; rebuild them per machine.
    """
    import tensorrt as trt
    import torch

    from app.models.tensorrt_backend import engine_metadata_path

    engine_path = models_dir / "yolov8n_fp16.engine"
    if not _should_build(engine_path, force):
        return engine_path

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    # The timing cache stores kernel benchmark results, so a rebuild on the
    # same GPU skips most of the (multi-minute) tactic search.
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


# ── Entry point ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
