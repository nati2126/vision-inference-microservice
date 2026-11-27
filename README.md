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
| pytorch | cpu | fp32 | 0.607 | 0.448 | 88.48 | 83.91 | 102.22 | 11.3 | 6.5 | 749 | n/a |
| pytorch | cuda | fp32 | 0.607 | 0.448 | 25.27 | 25.47 | 27.13 | 39.6 | 6.5 | 1377 | 357 |
| onnxruntime | cpu | fp32 | 0.594 | 0.444 | 51.16 | 48.88 | 66.31 | 19.5 | 12.7 | 170 | n/a |
| onnxruntime | cpu | int8 | 0.581 | 0.430 | 58.30 | 58.00 | 62.68 | 17.2 | 3.5 | 140 | n/a |
| onnxruntime | cuda | fp32 | 0.594 | 0.443 | 23.22 | 22.59 | 30.05 | 43.1 | 12.7 | 1156 | 221 |
| openvino | cpu | fp32 | 0.594 | 0.444 | 53.70 | 53.24 | 60.95 | 18.6 | 12.9 | 204 | n/a |
| openvino | cpu | int8 | 0.577 | 0.430 | 44.86 | 45.08 | 54.77 | 22.3 | 4.3 | 195 | n/a |
| tensorrt | cuda | fp16 | 0.595 | 0.444 | 15.96 | 15.82 | 18.75 | 62.7 | 9.4 | 806 | 242 |
| onnxruntime (ablation: naive INT8) | cpu | int8 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 3.5 | 129 | n/a |

The ONNX and OpenVINO artifacts take a dynamic batch. At batch 1 that costs a
little: run back to back against the earlier fixed-batch exports, `predict()`
was 0–9% slower (ONNX Runtime CPU 43.4 → 43.4–44.3 ms p50). The TensorRT
engine takes one image per call and is built from a static export, which keeps
it level with the earlier engine (see Design decisions).

These numbers move between runs on this laptop. Right after this run the CPU
was 18.5% busy with other programs (mostly a web browser), and every CPU row is
slower than in the run before it (ONNX Runtime FP32: 48.9 vs 42.4 ms p50).
Compare rows within one table; the interleaved A/Bs linked below are the
better evidence for any single change.

### Batching

The model call alone (`infer_batch`, no pre/post-processing) at batch 1 and 8.
ONNX Runtime on CUDA and the default TensorRT engine take batch 1 (see Design
decisions), and PyTorch's call is ultralytics' whole pipeline, which the
service runs one image at a time.

| Backend | Device | Precision | Max batch | batch 1 ms | batch 1 img/s | batch 8 ms | batch 8 img/s |
|---|---|---|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 1 | 102.55 | 9.8 | n/a | n/a |
| pytorch | cuda | fp32 | 1 | 31.20 | 32.1 | n/a | n/a |
| onnxruntime | cpu | fp32 | any | 38.04 | 26.3 | 238.22 | 33.6 |
| onnxruntime | cpu | int8 | any | 51.62 | 19.4 | 347.67 | 23.0 |
| onnxruntime | cuda | fp32 | 1 | 8.96 | 111.6 | n/a | n/a |
| openvino | cpu | fp32 | any | 47.15 | 21.2 | 271.87 | 29.4 |
| openvino | cpu | int8 | any | 32.29 | 31.0 | 227.12 | 35.2 |
| tensorrt | cuda | fp16 | 1 | 4.02 | 248.5 | n/a | n/a |

### HTTP

200 `POST /api/v1/detect` requests (the same 640x480 JPEG) per concurrency
level, uvicorn with one worker, httpx async client on the same machine.
`Batch max` is the server's `BATCH_MAX_SIZE` (1 = batching off), capped by
what the artifact accepts. The change per backend, and an interleaved
comparison against the previous serving code:
[`benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md`](benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md).
The step before that, narrowing the inference lock to the model call:
[`benchmark/results/i9-14900hx_rtx-4070-laptop-serving-ab.md`](benchmark/results/i9-14900hx_rtx-4070-laptop-serving-ab.md).

| Backend | Device | Precision | Batch max | c=1 p50 ms | c=1 p95 ms | c=1 req/s | c=8 p50 ms | c=8 p95 ms | c=8 req/s | c=32 p50 ms | c=32 p95 ms | c=32 req/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 1 | 114.1 | 121.2 | 8.7 | 833.0 | 856.6 | 9.5 | 3341.0 | 4576.6 | 9.0 |
| pytorch | cuda | fp32 | 1 | 37.4 | 43.4 | 26.4 | 245.7 | 809.7 | 23.4 | 977.4 | 1103.6 | 30.7 |
| onnxruntime | cpu | fp32 | 1 | 55.0 | 61.9 | 17.9 | 303.1 | 324.0 | 26.1 | 1203.7 | 1367.9 | 25.6 |
| onnxruntime | cpu | fp32 | 8 | 55.0 | 60.0 | 18.0 | 267.2 | 368.7 | 27.8 | 1134.0 | 1230.1 | 27.7 |
| onnxruntime | cpu | int8 | 1 | 67.3 | 73.5 | 14.7 | 398.1 | 429.8 | 19.9 | 1579.1 | 1680.6 | 19.9 |
| onnxruntime | cpu | int8 | 8 | 66.9 | 70.9 | 14.8 | 361.0 | 379.1 | 22.0 | 1437.4 | 1470.0 | 22.0 |
| onnxruntime | cuda | fp32 | 1 | 29.0 | 37.2 | 33.6 | 106.8 | 123.0 | 73.8 | 484.4 | 552.0 | 63.5 |
| openvino | cpu | fp32 | 1 | 67.6 | 75.8 | 14.7 | 371.8 | 406.0 | 21.4 | 1505.3 | 1593.8 | 20.9 |
| openvino | cpu | fp32 | 8 | 68.8 | 78.1 | 14.4 | 313.6 | 374.2 | 25.0 | 1186.9 | 1320.3 | 26.0 |
| openvino | cpu | int8 | 1 | 50.4 | 58.7 | 19.9 | 268.3 | 309.9 | 29.4 | 1088.2 | 1160.2 | 29.3 |
| openvino | cpu | int8 | 8 | 51.2 | 61.1 | 19.4 | 263.7 | 304.8 | 29.9 | 946.7 | 1034.5 | 33.3 |
| tensorrt | cuda | fp16 | 1 | 24.5 | 28.0 | 40.7 | 73.8 | 78.3 | 106.6 | 288.9 | 368.6 | 98.1 |

### Reproduce

```bash
pip install -e ".[openvino,tensorrt,bench]"   # whichever extras fit your machine
python -m scripts.export_models                 # first TensorRT build: ~8 min
python -m scripts.export_models --trt-max-batch 16   # a TensorRT engine that batches
python -m benchmark.run_benchmark               # full: ~11 min here
python -m benchmark.run_benchmark --quick       # smoke run: ~3 min here
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
of about 2.5, so every score rounds to 0. Keeping the decode nodes after the
head's convolutions in FP32 (124 in this export, most of them shape arithmetic
for the dynamic batch) recovers **0.430** against 0.444 for FP32; 63 of
the 64 convolutions are still INT8 (the exception is the DFL's fixed-weight
1x1 conv, which is part of the decode tail). `--int8-naive` on the export script
reproduces the broken file, and the benchmark scores it as an ablation.

**One pre/post-processing implementation for all exported backends**, so
accuracy differences come from runtimes and precisions, not from three copies
of the decode code. On the same ONNX file it matches ultralytics bit for bit.

**Trade-offs actually observed (tables above):**

- **TensorRT FP16 has the fastest model call:** 4.0 ms against 9.0 ms for ONNX
  Runtime on CUDA, with no accuracy loss (0.444 vs 0.444 for FP32 ONNX). End to
  end it is 15.8 ms p50, 5.3x PyTorch on the CPU and 1.6x PyTorch on CUDA.
  Most of those 15.8 ms are not the network: `predict()` letterboxes (about
  5 ms) and decodes plus NMS (about 3 ms) in NumPy on the CPU.
- **Leaving PyTorch is the biggest CPU win before any quantisation:** ONNX
  Runtime FP32 runs at 48.9 ms vs 83.9 ms, with under a quarter of the memory
  (170 vs 749 MB peak RSS).
- **INT8 is a 3.6x smaller file for −0.014 mAP50-95, but the speed-up depends on
  the runtime.** OpenVINO INT8 is 1.2x faster than OpenVINO FP32 (45.1 vs
  53.2 ms). ONNX Runtime INT8 is *slower* than ONNX Runtime FP32 on this CPU
  (58.0 vs 48.9 ms). Timing `session.run` alone shows the same, so it is not
  the pre/post-processing. I have not established the cause: candidates are
  the extra Q/DQ transitions around the FP32 tail and ORT's INT8 kernel choice
  on this hybrid P/E-core CPU.
- **PyTorch scores slightly higher (0.448 vs 0.444)** because ultralytics
  letterboxes `.pt` models to a minimal rectangle (640x480 for a 4:3 image)
  instead of the exported models' fixed 640x640, so its inputs differ a little.
- **Opportunistic batching, never a wait.** One worker task runs the model.
  It takes the next queued request plus any others already waiting, up to
  `BATCH_MAX_SIZE`, and makes one `infer_batch` call. Decoding, letterboxing
  and NMS run in the thread pool alongside it. With `BATCH_TIMEOUT_MS=0`, a
  request that arrives while the model is idle runs at once; batches only form
  when requests queue up behind a running call. Every batch size is run once at
  startup, because OpenVINO prepares each new input shape on first use (about
  80–100 ms per shape here). When the model takes only batch 1, a lock
  serialises the call instead: the worker cost about 5% there.
- **Batching pays off where the model call dominates: the CPU backends.** In
  an interleaved A/B, ONNX Runtime FP32 on the CPU served 16–23% more requests
  than the previous lock-based server at concurrency 8 and 32. Batching off vs
  on in full runs ranges from +7% to +28% for it, depending on how busy the
  machine was.
- **TensorRT does not batch by default.** A batch-16 engine (`--trt-max-batch
  16`) gave no clear gain: one server process is bound by its CPU work
  (decode, letterbox, NMS, HTTP) near 120–150 req/s, not by the 4 ms model
  call. It also cost about 5% at batch 1 and 594 MB of GPU memory instead of
  242 MB. The default engine matches the previous server in an interleaved
  A/B. Details:
  [`benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md`](benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md).
- **ONNX Runtime on CUDA does not batch.** Its CUDA provider re-plans every
  convolution when the input shape changes, about 100–130 ms per change here,
  and neither `cudnn_conv_algo_search` nor the memory-arena options avoided it.
  With varying batch sizes, throughput at concurrency 8 fell from 59.8 to
  27.0 req/s, so the backend reports a batch limit of 1. TensorRT pays about
  3 ms per change; ONNX Runtime on the CPU and OpenVINO pay nothing measurable.
- **HTTP adds about 6–10 ms** per request over `predict()` at concurrency 1
  (ONNX Runtime CPU: 55.0 vs 48.9 ms p50 here, 52.3 vs 42.4 in the run before).

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
- **One process, one model.** Batching only forms under concurrency, and for
  the GPU backends the single uvicorn process runs out of CPU before the GPU is
  busy. More workers would each load their own copy of the model. The default
  TensorRT engine and ONNX Runtime on CUDA serve one image per call (see Design
  decisions).
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
  "device": "cpu",
  "max_batch_size": 8
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
| `BATCH_MAX_SIZE` | `8` | Most queued requests run as one model call; `1` turns batching off. Capped by the model (`max_batch_size` in `/health`) |
| `BATCH_TIMEOUT_MS` | `0` | How long a model call waits for more requests. `0` only takes those already queued, so a lone request is never delayed |
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
