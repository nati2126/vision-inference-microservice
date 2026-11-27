# API and configuration

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

## `GET /api/v1/health`

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

## `GET /livez` and `GET /readyz`

Probes for orchestrators, outside the versioned API. `/livez` returns 200
`{"status": "ok"}` while the process serves requests. `/readyz` returns 200
`{"status": "ready"}` once the model is loaded and 503 `{"status": "not_ready"}`
otherwise; the Docker and Compose health checks use it.

## `GET /metrics`

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

## `POST /api/v1/detect`

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

## Errors

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

## Configuration

All settings come from environment variables (see [`.env.example`](../.env.example)):

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
