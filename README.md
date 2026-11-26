# Vision Inference Microservice

[![CI](https://github.com/nati2126/vision-inference-microservice/actions/workflows/ci.yml/badge.svg)](https://github.com/nati2126/vision-inference-microservice/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

A YOLOv8 object-detection service built on FastAPI. The same HTTP API can run
on **PyTorch, ONNX Runtime (CPU/CUDA, FP32/INT8), OpenVINO (FP32/INT8) or
TensorRT (FP16)**, selected by an environment variable. A benchmark measures
every backend on the same machine with the same evaluation code.

The exported backends run only the network; letterboxing, YOLOv8 output
decoding, NMS and box rescaling are written out in NumPy
([`app/models/processing.py`](app/models/processing.py)), and verified to match
ultralytics' own pipeline exactly on the same ONNX file.

![Detections returned by POST /api/v1/detect on the TensorRT FP16 backend](docs/demo.jpg)

*Boxes drawn from the JSON returned by `POST /api/v1/detect`, with the service
running the TensorRT FP16 engine. Regenerate against any running backend with
`python -m scripts.render_demo path/to/a.jpg path/to/b.jpg --url http://localhost:8000`.*

---

## Architecture

```
app/
├── api/v1/endpoints/     # Thin route handlers (API layer)
├── schemas/              # Pydantic request / response models
├── services/             # Upload handling, decoding, inference orchestration
├── models/               # InferenceBackend interface + one module per runtime
│   ├── base.py           #   the interface and the Detection record
│   ├── processing.py     #   letterbox, decode, NMS, rescale (NumPy)
│   ├── factory.py        #   MODEL_BACKEND -> backend (lazy imports)
│   └── *_backend.py      #   pytorch, onnxruntime, openvino, tensorrt
└── core/                 # Config, logging
scripts/export_models.py  # .pt -> ONNX FP32 / INT8, OpenVINO IR, TensorRT engine
benchmark/                # accuracy, latency, memory and HTTP benchmark
```

| Layer | Responsibility |
|-------|---------------|
| **API** | HTTP routing, input validation, error mapping |
| **Service** | Size-capped upload, image decoding, off-loop inference, metadata |
| **Model** | `InferenceBackend`: `load()`, `predict(image) -> list[Detection]`, `name`, `precision` |
| **Core** | Configuration (12-factor), structured logging |

---

## Quick start

```bash
git clone https://github.com/nati2126/vision-inference-microservice.git
cd vision-inference-microservice
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # PyTorch + ONNX Runtime (CPU)
cp .env.example .env
uvicorn app.main:app --reload      # PyTorch backend, CPU
```

Serve another backend:

```bash
python -m scripts.export_models    # writes models/ (gitignored), prints sizes
MODEL_BACKEND=onnxruntime MODEL_PATH=models/yolov8n_int8.onnx uvicorn app.main:app
```

Docker: `make docker-up` (PyTorch backend, CPU).

---

## Backends

| `MODEL_BACKEND` | `MODEL_DEVICE` | Precision | `MODEL_PATH` | Install |
|---|---|---|---|---|
| `pytorch` | `cpu`, `cuda`, `mps` | FP32 | `yolov8n.pt` (downloaded) | base |
| `onnxruntime` | `cpu` | FP32 | `models/yolov8n.onnx` | base |
| `onnxruntime` | `cpu` | INT8 | `models/yolov8n_int8.onnx` | base |
| `onnxruntime` | `cuda` | FP32 | `models/yolov8n.onnx` | `[gpu]` (see note) |
| `openvino` | `cpu` | FP32 | `models/yolov8n_openvino/yolov8n.xml` | `[openvino]` |
| `openvino` | `cpu` | INT8 | `models/yolov8n_int8_openvino/yolov8n_int8.xml` | `[openvino]` |
| `tensorrt` | `cuda` | FP16 | `models/yolov8n_fp16.engine` | `[tensorrt]` |

Extras are declared in `pyproject.toml`: `pip install -e ".[openvino,tensorrt,bench]"`.
`onnxruntime` and `onnxruntime-gpu` install the same module and cannot coexist:
for CUDA, `pip uninstall onnxruntime` and then install the `[gpu]` extra.

The precision reported by `/health` is read from the artifact (QDQ nodes in the
ONNX graph, the compiled OpenVINO model, the engine's metadata), not from
configuration. A backend asked for CUDA that cannot get it fails at startup
instead of silently running on the CPU.

---

## Benchmarks

**Hardware:** Intel Core i9-14900HX (24 cores: 8P + 16E, 32 threads), 32 GB RAM,
NVIDIA GeForce RTX 4070 Laptop GPU (8 GB, driver 591.86), Windows 11, on AC power.
Python 3.11, torch 2.6.0+cu124, ultralytics 8.3.57, onnxruntime-gpu 1.20.2,
OpenVINO 2026.4.1, TensorRT 10.16.1. yolov8n, 640x640 input.
Full output, including every package version and the commit:
[`benchmark/results/i9-14900hx_rtx-4070-laptop.md`](benchmark/results/i9-14900hx_rtx-4070-laptop.md)
(and `.json`).

### Model

Accuracy on **coco128** (128 images, 929 boxes), confidence 0.001, NMS IoU 0.7,
identical evaluation code for every backend. coco128 is a small, quick set:
these numbers compare the backends with each other and are **not** official
COCO val2017 results. Latency is `predict()` end to end (preprocess + inference
+ postprocess), batch 1, 10 warm-up + 100 timed runs on a 640x480 image.

| Backend | Device | Precision | mAP50 | mAP50-95 | Mean ms | p50 ms | p95 ms | img/s | Size MB | Peak RSS MB | Peak GPU MB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 0.607 | 0.448 | 92.61 | 91.65 | 100.75 | 10.8 | 6.5 | 751 | n/a |
| pytorch | cuda | fp32 | 0.607 | 0.448 | 22.77 | 22.48 | 25.65 | 43.9 | 6.5 | 1379 | 357 |
| onnxruntime | cpu | fp32 | 0.594 | 0.444 | 40.44 | 40.60 | 42.27 | 24.7 | 12.8 | 172 | n/a |
| onnxruntime | cpu | int8 | 0.581 | 0.430 | 51.36 | 52.37 | 53.28 | 19.5 | 3.7 | 142 | n/a |
| onnxruntime | cuda | fp32 | 0.594 | 0.443 | 18.99 | 18.79 | 20.52 | 52.7 | 12.8 | 1152 | 221 |
| openvino | cpu | fp32 | 0.594 | 0.444 | 45.33 | 44.55 | 54.26 | 22.1 | 12.9 | 182 | n/a |
| openvino | cpu | int8 | 0.577 | 0.430 | 32.17 | 30.09 | 40.67 | 31.1 | 4.4 | 209 | n/a |
| tensorrt | cuda | fp16 | 0.595 | 0.444 | 13.84 | 13.92 | 15.00 | 72.2 | 9.4 | 807 | 242 |
| onnxruntime (ablation: naive INT8) | cpu | int8 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 3.6 | 128 | n/a |

### HTTP

200 `POST /api/v1/detect` requests (the same 640x480 JPEG) per concurrency
level, uvicorn with one worker, httpx async client on the same machine.
Re-measured on a later day than the model table, after the inference lock was
narrowed to the model call alone; compare rows within one table, not across
the two. Before/after for that change:
[`benchmark/results/i9-14900hx_rtx-4070-laptop-serving-ab.md`](benchmark/results/i9-14900hx_rtx-4070-laptop-serving-ab.md).

| Backend | Device | Precision | c=1 p50 ms | c=1 p95 ms | c=1 req/s | c=8 p50 ms | c=8 p95 ms | c=8 req/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 81.2 | 109.2 | 11.5 | 739.6 | 820.3 | 11.4 |
| pytorch | cuda | fp32 | 29.5 | 43.2 | 33.9 | 164.2 | 290.0 | 43.7 |
| onnxruntime | cpu | fp32 | 52.4 | 54.4 | 19.1 | 276.5 | 284.7 | 28.7 |
| onnxruntime | cpu | int8 | 63.3 | 64.6 | 15.8 | 363.2 | 365.7 | 22.0 |
| onnxruntime | cuda | fp32 | 23.8 | 32.9 | 38.2 | 73.3 | 100.4 | 101.5 |
| openvino | cpu | fp32 | 62.2 | 70.6 | 15.9 | 288.7 | 383.3 | 26.7 |
| openvino | cpu | int8 | 41.7 | 50.0 | 23.7 | 231.1 | 257.7 | 34.3 |
| tensorrt | cuda | fp16 | 19.6 | 24.6 | 48.3 | 53.4 | 65.4 | 145.4 |

### Reproduce

```bash
pip install -e ".[openvino,tensorrt,bench]"   # whichever extras fit your machine
python -m scripts.export_models                 # first TensorRT build: ~8 min
python -m benchmark.run_benchmark               # full: ~6.5 min here
python -m benchmark.run_benchmark --quick       # smoke run: ~2.5 min here
```

Backends whose package, artifact or GPU is missing are listed as skipped. Each
backend's model measurements run in a fresh subprocess so peak memory is its
own. Results are written to `benchmark/results/<hardware-tag>.{json,md}`.

---

## Design decisions

**Letterbox, not resize.** The network takes a fixed 640x640 input. Squashing a
4:3 photo into a square distorts every object, so the image is scaled by one
factor until its long side is 640 and the short side is padded with grey (114,
as in training). Keeping the scale and the padding offsets makes the inverse
exact: subtract the padding, divide by the scale, clip. A unit test round-trips
boxes through both directions.

**Static INT8 quantisation, with the detect head's tail left in FP32.** ONNX
Runtime's static quantiser measures activation ranges once, on 100 coco128
images preprocessed exactly as at inference time, and bakes them into the graph
as QuantizeLinear/DequantizeLinear (QDQ) pairs: per-channel INT8 weights, UInt8
activations. Quantising the whole graph scores **mAP50-95 0.000**.
YOLOv8's final Concat joins box coordinates (0–640 px) with class
scores (0–1) in one tensor, and a single UInt8 scale covering 0–640 has a step
of about 2.5, so every score rounds to 0. Keeping the 24 decode nodes after the
head's convolutions in FP32 recovers **0.430** against 0.444 for FP32; 63 of
the 64 convolutions are still INT8 (the exception is the DFL's fixed-weight
1x1 conv, which is part of the decode tail). `--int8-naive` on the export script
reproduces the broken file, and the benchmark scores it as an ablation.

**One pre/post-processing implementation for all exported backends**, so
accuracy differences come from runtimes and precisions, not from three copies
of the decode code. On the same ONNX file it matches ultralytics bit for bit.

**Trade-offs actually observed (table above):**

- **TensorRT FP16 is the fastest:** 13.8 ms (6.7x PyTorch on CPU, 1.6x PyTorch
  on CUDA) with no accuracy loss (0.444 vs 0.444 for FP32 ONNX).
- **Leaving PyTorch is the biggest CPU win before any quantisation:** ONNX
  Runtime FP32 runs at 40.4 ms vs 92.6 ms, with about a quarter of the memory
  (172 vs 751 MB peak RSS).
- **INT8 is a 3.5x smaller file for −0.014 mAP50-95, but the speed-up depends on
  the runtime.** OpenVINO INT8 is 1.4x faster than OpenVINO FP32 (32.2 vs
  45.3 ms). ONNX Runtime INT8 is *slower* than ONNX Runtime FP32 on this CPU
  (51.4 vs 40.4 ms). Timing `session.run` alone shows the same, so it is not
  the pre/post-processing. I have not established the cause: candidates are
  the extra Q/DQ transitions around the FP32 tail and ORT's INT8 kernel choice
  on this hybrid P/E-core CPU.
- **PyTorch scores slightly higher (0.448 vs 0.444)** because ultralytics
  letterboxes `.pt` models to a minimal rectangle (640x480 for a 4:3 image)
  instead of the exported models' fixed 640x640, so its inputs differ a little.
- **Only the model call is serialised.** There is one model instance per
  process, and a lock lets one request at a time into it. Decoding,
  letterboxing and NMS for other requests run alongside it in the thread pool.
  Going from concurrency 1 to 8 lifts throughput 2.7–3.0x on the GPU backends
  and 1.4–1.7x on the CPU ones, where those stages compete with inference for
  cores. PyTorch gains little (1.0–1.3x), because ultralytics letterboxes and
  runs NMS inside the locked call. When the lock covered all three stages,
  TensorRT served 72.5 req/s at concurrency 8; it now serves 145.4 (A/B run
  back to back on the same day). Going further needs dynamic batching or more
  workers.
- **HTTP adds about 10 ms** per request over `predict()` at concurrency 1
  (ONNX Runtime CPU in the original run: 50.3 vs 40.6 ms p50).

**Two runtime traps, both handled in code.** ONNX Runtime silently falls back
to the CPU when the CUDA provider fails to initialise; the backend checks the
session's providers and refuses to start. Given `device="cpu"`, ultralytics
sets `CUDA_VISIBLE_DEVICES=-1` for the whole process, which hides the GPU from
every other CUDA library loaded afterwards; the PyTorch backend passes a
`torch.device` instead, which skips that code path.

---

## Limitations

- **Small evaluation set.** coco128 has 128 images and 929 boxes; differences
  of around 0.01 mAP are within what a different 128 images could change. The
  INT8 calibration images (100) come from the same set; calibration uses no
  labels, only activation ranges, but the two are not independent.
- **One machine.** A laptop with a hybrid P/E-core CPU, Windows, mains power.
  Thread placement, thermals and power plans move CPU numbers; re-run the
  benchmark on your hardware rather than transferring these numbers.
- **Batch size 1 only.** No dynamic batching; GPU utilisation is far from
  saturated at these latencies.
- **TensorRT engines are not portable.** They are tied to the GPU model and
  TensorRT version; the first build takes about 8 minutes here (a timing
  cache speeds up rebuilds).
- **Memory figures are approximate.** GPU memory is NVML's device-level usage
  minus the level before loading. CUDA rows also include the CUDA context
  and, for ONNX Runtime on CUDA, importing torch to load the CUDA libraries.
- **The HTTP client shares the machine** with the server and takes CPU from it.
- **CI covers CPU only** (PyTorch and ONNX Runtime). OpenVINO, TensorRT and the
  CUDA paths are tested locally; their tests skip on the CI runner.
- **No TensorRT INT8 and no INT8 on CUDA**; INT8 here is CPU-only.

---

## API reference

### `GET /api/v1/health`

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true,
  "backend": "onnxruntime",
  "precision": "int8",
  "device": "cpu"
}
```

### `GET /livez` and `GET /readyz`

Probes for orchestrators, outside the versioned API. `/livez` returns 200
`{"status": "ok"}` while the process serves requests. `/readyz` returns 200
`{"status": "ready"}` once the model is loaded and 503 `{"status": "not_ready"}`
otherwise; the Docker and Compose health checks use it.

### `POST /api/v1/detect`

```bash
curl -X POST http://localhost:8000/api/v1/detect -F "file=@image.jpg"
```

```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.9231,
      "bbox": {"x_min": 45.12, "y_min": 102.34, "x_max": 320.56, "y_max": 489.78}
    }
  ],
  "metadata": {
    "image_width": 640,
    "image_height": 480,
    "inference_time_ms": 42.15,
    "detections_count": 1
  }
}
```

The response is identical for every backend. `inference_time_ms` covers
preprocessing, the model call and postprocessing, but not time spent waiting
for the model while other requests use it.

### Errors

Failures return the `ErrorResponse` envelope:

```json
{
  "error": "Unsupported file type 'text/plain'. Accepted: image/bmp, image/jpeg, ...",
  "detail": null
}
```

| Status | Cause |
|--------|-------|
| `400` | Unsupported content type, undecodable image, empty body, or over `MAX_UPLOAD_BYTES` / `MAX_IMAGE_PIXELS` |
| `422` | Malformed request (e.g. no file field) |
| `500` | Inference failure: details are logged, never returned |

Interactive docs: [`/docs`](http://localhost:8000/docs) (Swagger UI),
[`/redoc`](http://localhost:8000/redoc).

---

## Configuration

All settings come from environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BACKEND` | `pytorch` | `pytorch`, `onnxruntime`, `openvino` or `tensorrt` |
| `MODEL_PATH` | `yolov8n.pt` | Artifact for the backend (`MODEL_NAME` is accepted as an alias) |
| `MODEL_DEVICE` | `cpu` | `cpu`, `cuda` or `mps` (PyTorch only) |
| `MODEL_CONFIDENCE_THRESHOLD` | `0.25` | Minimum detection confidence |
| `MODEL_IOU_THRESHOLD` | `0.7` | NMS overlap threshold |
| `MODEL_MAX_DETECTIONS` | `300` | Maximum detections per image |
| `MODEL_WEIGHTS_DIR` | `~/.cache/vision-inference/weights` | Where a bare `.pt` filename is downloaded and cached |
| `MAX_UPLOAD_BYTES` | `10485760` | Largest accepted upload (10 MiB) |
| `MAX_IMAGE_PIXELS` | `50000000` | Largest accepted width x height, read from the image header before decoding |
| `CORS_ALLOW_ORIGINS` | `["*"]` | Allowed origins, as a JSON list |
| `CORS_ALLOW_CREDENTIALS` | `false` | Ignored while origins is `["*"]`, since browsers reject that pairing |
| `LOG_LEVEL` | `INFO` | Logging severity |
| `LOG_FORMAT` | `json` | Output format (`json` / `console`) |
| `ENVIRONMENT` | `development` | Runtime environment |

---

## Development

```bash
make lint             # ruff
make typecheck        # mypy --strict
make test             # pytest
make export           # python -m scripts.export_models
make benchmark        # full benchmark
make benchmark-quick  # smoke benchmark
```

The tests export `models/yolov8n.{pt,onnx}` on first run. Tests for backends
whose package, artifact or GPU is missing are skipped with the reason.

---

## License

MIT
