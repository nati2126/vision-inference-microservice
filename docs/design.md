# Design

The exported backends run only the network; letterboxing, YOLOv8 output
decoding, NMS and box rescaling are written out in NumPy
([`app/models/processing.py`](../app/models/processing.py)), and verified to match
ultralytics' own pipeline exactly on the same ONNX file. TensorRT runs the
heavy parts of that pipeline on the GPU
([`app/models/gpu_processing.py`](../app/models/gpu_processing.py)), tested to give
a bit-identical input tensor and the same detections.

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

**Trade-offs actually observed (tables in [benchmarks](benchmarks.md)):**

- **TensorRT FP16 has the fastest model call:** 2.2 ms against 7.9 ms for ONNX
  Runtime on CUDA, with no accuracy loss (COCO val2017 mAP50-95 0.367 vs 0.367
  for FP32 ONNX). End to
  end it is 4.0 ms p50, 18x PyTorch on the CPU and 3.6x PyTorch on CUDA.
- **For TensorRT, the CPU pipeline was most of the time.** With the NumPy
  pre/post-processing, `predict()` took about 11 ms, for a model call of about
  3 ms. Most of that was converting the image to float32 (4.9 ms, against
  0.7 ms for the resize and padding). The backend now uploads the uint8 image,
  pads and scales it on the GPU, and copies back only the boxes above the
  confidence threshold. In an interleaved A/B, `predict()` p50 fell from
  11.0–11.2 to 4.0–5.1 ms, and HTTP throughput rose 53–65% at concurrency 1
  and 24–36% at 8 and 32
  ([`benchmark/results/i9-14900hx_rtx-4070-laptop-gpu-processing.md`](../benchmark/results/i9-14900hx_rtx-4070-laptop-gpu-processing.md)).
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
- **INT8 is a 3.6x smaller file for −0.011 mAP50-95 on val2017, but the
  speed-up depends on the runtime.** OpenVINO INT8 is 1.35x faster than OpenVINO FP32 (33.7 vs
  45.5 ms). ONNX Runtime INT8 is *slower* than ONNX Runtime FP32 on this CPU
  (51.3 vs 40.9 ms). Timing `session.run` alone shows the same, so it is not
  the pre/post-processing. I have not established the cause: candidates are
  the extra Q/DQ transitions around the FP32 tail and ORT's INT8 kernel choice
  on this hybrid P/E-core CPU.
- **The input shape does not change accuracy.** ultralytics letterboxes `.pt`
  models to a minimal rectangle (640x480 for a 4:3 image) instead of the
  exported models' fixed 640x640. On coco128 PyTorch scored 0.448 against
  0.444; on val2017 both score 0.367, so that gap was the small set's noise.
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
  [`benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md`](../benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md).
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
  ([`benchmark/results/i9-14900hx_rtx-4070-laptop-load-shedding.md`](../benchmark/results/i9-14900hx_rtx-4070-laptop-load-shedding.md)).
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

## Limitations

- **coco128 is small.** It has 128 images and 929 boxes; differences of around
  0.01 mAP are within what a different 128 images could change (PyTorch's
  +0.004 there vanished on val2017). The INT8 calibration images (100) come
  from it too. The full val2017 run takes about an hour here, so the main
  benchmark keeps coco128 and val2017 is a separate run.
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
