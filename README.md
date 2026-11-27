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
ultralytics' own pipeline exactly on the same ONNX file. TensorRT runs the
heavy parts of that pipeline on the GPU
([`app/models/gpu_processing.py`](app/models/gpu_processing.py)), tested to give
a bit-identical input tensor and the same detections.

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
│   ├── gpu_processing.py #   TensorRT: pad, scale and filter on the GPU (torch)
│   ├── factory.py        #   MODEL_BACKEND -> backend (lazy imports)
│   └── *_backend.py      #   pytorch, onnxruntime, openvino, tensorrt
└── core/                 # Config, logging, Prometheus metrics, load shedding
scripts/export_models.py  # .pt -> ONNX FP32 / INT8, OpenVINO IR, TensorRT engine
benchmark/                # accuracy, latency, memory and HTTP benchmark; overload test
```

| Layer | Responsibility |
|-------|---------------|
| **API** | HTTP routing, input validation, error mapping |
| **Service** | Size-capped upload, image decoding, off-loop inference, metadata |
| **Model** | `InferenceBackend`: `load()`, `predict(image) -> list[Detection]`, `name`, `precision` |
| **Core** | Configuration (12-factor), structured logging, metrics, admission control |

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
| pytorch | cpu | fp32 | 0.607 | 0.448 | 73.64 | 73.00 | 81.33 | 13.6 | 6.5 | 749 | n/a |
| pytorch | cuda | fp32 | 0.607 | 0.448 | 14.41 | 14.38 | 15.30 | 69.4 | 6.5 | 1377 | 357 |
| onnxruntime | cpu | fp32 | 0.594 | 0.444 | 40.87 | 40.91 | 42.96 | 24.5 | 12.7 | 172 | n/a |
| onnxruntime | cpu | int8 | 0.581 | 0.430 | 50.96 | 51.32 | 53.79 | 19.6 | 3.5 | 141 | n/a |
| onnxruntime | cuda | fp32 | 0.594 | 0.443 | 15.91 | 15.94 | 16.55 | 62.9 | 12.7 | 1161 | 221 |
| openvino | cpu | fp32 | 0.594 | 0.444 | 45.98 | 45.45 | 53.18 | 21.7 | 12.9 | 205 | n/a |
| openvino | cpu | int8 | 0.577 | 0.430 | 32.39 | 33.72 | 37.14 | 30.9 | 4.3 | 195 | n/a |
| tensorrt | cuda | fp16 | 0.595 | 0.444 | 4.00 | 3.98 | 4.28 | 249.8 | 9.4 | 912 | 307 |
| onnxruntime (ablation: naive INT8) | cpu | int8 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 3.5 | 130 | n/a |

The ONNX and OpenVINO artifacts take a dynamic batch. At batch 1 that costs a
little: run back to back against the earlier fixed-batch exports, `predict()`
was 0–9% slower (ONNX Runtime CPU 43.4 → 43.4–44.3 ms p50). The TensorRT
engine takes one image per call and is built from a static export, which keeps
it level with the earlier engine (see Design decisions).

These numbers move between runs on this laptop. Before this run the CPU was
6.5% busy with other programs; in the run before it, 18.5% (mostly a web
browser), and every CPU row was slower (ONNX Runtime FP32: 48.9 vs 40.9 ms
p50). Compare rows within one table; the interleaved A/Bs linked below are the
better evidence for any single change.

### Batching

The model call alone (`infer_batch`, no pre/post-processing) at batch 1 and 8.
ONNX Runtime on CUDA and the default TensorRT engine take batch 1 (see Design
decisions), and PyTorch's call is ultralytics' whole pipeline, which the
service runs one image at a time.

| Backend | Device | Precision | Max batch | batch 1 ms | batch 1 img/s | batch 8 ms | batch 8 img/s |
|---|---|---|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 1 | 75.47 | 13.3 | n/a | n/a |
| pytorch | cuda | fp32 | 1 | 14.97 | 66.8 | n/a | n/a |
| onnxruntime | cpu | fp32 | any | 32.78 | 30.5 | 199.72 | 40.1 |
| onnxruntime | cpu | int8 | any | 43.03 | 23.2 | 305.66 | 26.2 |
| onnxruntime | cuda | fp32 | 1 | 7.92 | 126.2 | n/a | n/a |
| openvino | cpu | fp32 | any | 34.31 | 29.2 | 219.47 | 36.5 |
| openvino | cpu | int8 | any | 22.52 | 44.4 | 172.04 | 46.5 |
| tensorrt | cuda | fp16 | 1 | 2.15 | 465.0 | n/a | n/a |

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
| pytorch | cpu | fp32 | 1 | 84.1 | 90.0 | 11.8 | 614.6 | 631.2 | 13.0 | 2453.2 | 2479.3 | 12.9 |
| pytorch | cuda | fp32 | 1 | 29.9 | 35.1 | 34.7 | 178.5 | 273.4 | 42.1 | 731.2 | 911.3 | 39.0 |
| onnxruntime | cpu | fp32 | 1 | 51.0 | 53.1 | 19.7 | 270.9 | 280.6 | 29.3 | 1083.0 | 1101.0 | 29.2 |
| onnxruntime | cpu | fp32 | 8 | 51.0 | 53.2 | 19.6 | 228.4 | 268.2 | 34.8 | 858.7 | 903.6 | 36.6 |
| onnxruntime | cpu | int8 | 1 | 63.1 | 65.1 | 16.0 | 359.0 | 368.4 | 22.2 | 1429.6 | 1444.7 | 22.2 |
| onnxruntime | cpu | int8 | 8 | 62.3 | 63.7 | 16.2 | 327.3 | 355.5 | 24.3 | 1294.7 | 1311.2 | 24.4 |
| onnxruntime | cuda | fp32 | 1 | 24.3 | 25.4 | 40.9 | 76.2 | 86.4 | 102.0 | 363.1 | 406.8 | 84.5 |
| openvino | cpu | fp32 | 1 | 58.3 | 66.8 | 17.0 | 299.8 | 337.6 | 26.2 | 1186.8 | 1249.8 | 26.7 |
| openvino | cpu | fp32 | 8 | 58.5 | 63.3 | 17.1 | 249.6 | 281.3 | 32.0 | 939.5 | 1010.3 | 33.0 |
| openvino | cpu | int8 | 1 | 37.7 | 45.3 | 25.6 | 187.3 | 234.8 | 40.9 | 749.5 | 854.0 | 40.9 |
| openvino | cpu | int8 | 8 | 37.8 | 45.5 | 25.5 | 207.2 | 246.1 | 39.3 | 773.4 | 889.6 | 41.2 |
| tensorrt | cuda | fp16 | 1 | 14.8 | 16.2 | 70.8 | 40.7 | 61.3 | 186.3 | 149.2 | 1037.1 | 123.9 |

### Reproduce

```bash
pip install -e ".[openvino,tensorrt,bench]"   # whichever extras fit your machine
python -m scripts.export_models                 # first TensorRT build: ~8 min
python -m scripts.export_models --trt-max-batch 16   # a TensorRT engine that batches
python -m benchmark.run_benchmark               # full: ~11 min here
python -m benchmark.run_benchmark --quick       # smoke run: ~3 min here
python -m benchmark.overload                    # load shedding under overload: ~4 min
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
TensorRT's GPU version reuses its geometry and its NMS, and only moves the
padding, scaling and confidence filter; tests hold it to a bit-identical input
tensor and the same detections.

**Trade-offs actually observed (tables above):**

- **TensorRT FP16 has the fastest model call:** 2.2 ms against 7.9 ms for ONNX
  Runtime on CUDA, with no accuracy loss (0.444 vs 0.444 for FP32 ONNX). End to
  end it is 4.0 ms p50, 18x PyTorch on the CPU and 3.6x PyTorch on CUDA.
- **For TensorRT, the CPU pipeline was most of the time.** With the NumPy
  pre/post-processing, `predict()` took about 11 ms, for a model call of about
  3 ms. Most of that was converting the image to float32 (4.9 ms, against
  0.7 ms for the resize and padding). The backend now uploads the uint8 image,
  pads and scales it on the GPU, and copies back only the boxes above the
  confidence threshold. In an interleaved A/B, `predict()` p50 fell from
  11.0–11.2 to 4.0–5.1 ms, and HTTP throughput rose 53–65% at concurrency 1
  and 24–36% at 8 and 32
  ([`benchmark/results/i9-14900hx_rtx-4070-laptop-gpu-processing.md`](benchmark/results/i9-14900hx_rtx-4070-laptop-gpu-processing.md)).
  It costs 65 MB of GPU memory (307 vs 242 MB). Two things stayed on the CPU on
  measurement. The OpenCV resize: torch's bilinear resize differed by a grey
  level on 9% of pixels and moved an FP16 box. And NMS: torchvision's GPU
  version was 5x slower for the ~50 boxes left at the default threshold.
- **The TensorRT HTTP row at c=32 has a tail** in this run (p95 1037 ms,
  123.9 req/s). The interleaved A/B measured 172–177 req/s at c=32 with the
  same code.
- **Leaving PyTorch is the biggest CPU win before any quantisation:** ONNX
  Runtime FP32 runs at 40.9 ms vs 73.0 ms, with under a quarter of the memory
  (172 vs 749 MB peak RSS).
- **INT8 is a 3.6x smaller file for −0.014 mAP50-95, but the speed-up depends on
  the runtime.** OpenVINO INT8 is 1.35x faster than OpenVINO FP32 (33.7 vs
  45.5 ms). ONNX Runtime INT8 is *slower* than ONNX Runtime FP32 on this CPU
  (51.3 vs 40.9 ms). Timing `session.run` alone shows the same, so it is not
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
  16`) gave no clear gain: one server process was bound by its CPU work
  (decode, letterbox, NMS, HTTP) near 120–150 req/s, not by the 4 ms model
  call. That was measured before pre/post-processing moved to the GPU, and
  has not been re-measured since. It also cost about 5% at batch 1 and 594 MB of GPU memory instead of
  242 MB. The default engine matches the previous server in an interleaved
  A/B. Details:
  [`benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md`](benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md).
- **ONNX Runtime on CUDA does not batch.** Its CUDA provider re-plans every
  convolution when the input shape changes, about 100–130 ms per change here,
  and neither `cudnn_conv_algo_search` nor the memory-arena options avoided it.
  With varying batch sizes, throughput at concurrency 8 fell from 59.8 to
  27.0 req/s, so the backend reports a batch limit of 1. TensorRT pays about
  3 ms per change; ONNX Runtime on the CPU and OpenVINO pay nothing measurable.
- **Overload is shed, not queued.** Past `MAX_IN_FLIGHT_REQUESTS` (64), the
  detect endpoint answers 503 with `Retry-After` before parsing the upload.
  With 256 clients against the ONNX Runtime CPU backend (about 30 req/s),
  served requests took 1.9 s p50 instead of 7.2 s, for 1–2% less goodput
  ([`benchmark/results/i9-14900hx_rtx-4070-laptop-load-shedding.md`](benchmark/results/i9-14900hx_rtx-4070-laptop-load-shedding.md)).
  The middleware reads and discards a rejected upload before answering: over
  a real socket, an unread upload left the client blocked mid-send and the
  request hung.
- **HTTP adds about 10 ms** per request over `predict()` at concurrency 1
  (ONNX Runtime CPU: 51.0 vs 40.9 ms p50; TensorRT: 14.8 vs 4.0 ms). For
  TensorRT that is now most of the request.

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
- **The in-flight limit is a count, not a deadline.** 64 requests is about 2 s
  of queue on a CPU backend and well under 1 s on TensorRT. Set
  `MAX_IN_FLIGHT_REQUESTS` from your backend's throughput and latency budget.
  There is no authentication or per-client rate limiting; put the service
  behind a gateway for that.
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

### `GET /metrics`

Prometheus text format:

| Metric | Meaning |
|---|---|
| `vision_detect_requests_total{status}` | Detect responses by HTTP status, including 503 rejections |
| `vision_detect_request_duration_seconds` | Histogram: accepted detect request to last response byte |
| `vision_detect_in_flight_requests` | Detect requests accepted and not yet answered |
| `vision_detect_rejected_total` | Requests turned away at `MAX_IN_FLIGHT_REQUESTS` |
| `vision_model_queue_wait_seconds` | Histogram: preprocessed request waiting for its model call |
| `vision_model_call_duration_seconds` | Histogram: one model call (a whole batch) |
| `vision_model_batch_size` | Histogram: requests served per model call |
| `vision_model_info{backend,device,precision,max_batch_size}` | The loaded model (always 1) |

Under load, queue wait rising while the model call stays flat means the
server is past capacity. Batch sizes near `BATCH_MAX_SIZE` mean batching is
doing its job.

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
| `503` | `MAX_IN_FLIGHT_REQUESTS` detect requests already in flight; carries `Retry-After: 1` |

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
| `MAX_IN_FLIGHT_REQUESTS` | `64` | Detect requests served at once; beyond it the server answers 503 with `Retry-After` instead of queueing. About 2 s of queue for a CPU backend at 30 req/s |
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
