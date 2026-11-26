# Serving path A/B: inference lock scope

Before: `ff4df49` (decode on the event loop; letterbox, model call and NMS all under the
inference lock). After: `4d21f97` (decode and pre/post-processing in the thread pool; only
the model call under the lock).

Same machine as `i9-14900hx_rtx-4070-laptop.md`, measured 2026-11-26. For each config the
before and after servers ran back to back, so both see the same thermal and power state.
200 `POST /api/v1/detect` requests per concurrency level after 5 warm-up requests, the same
640x480 coco128 JPEG, uvicorn with one worker, httpx async client on the same machine.
No request failed. Each run called `benchmark.http_load.run_http_benchmark` from a checkout of
the commit under test. Raw data: the `.json` file next to this one.

| Config | c=1 req/s before | c=1 req/s after | c=8 p50 ms before | c=8 p50 ms after | c=8 p95 ms before | c=8 p95 ms after | c=8 req/s before | c=8 req/s after | c=8 change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch-cpu-fp32 | 9.4 | 11.5 | 752.44 | 739.57 | 809.06 | 820.28 | 10.5 | 11.4 | +9% |
| pytorch-cuda-fp32 | 30.0 | 33.9 | 191.53 | 164.17 | 272.46 | 289.98 | 40.3 | 43.7 | +8% |
| onnxruntime-cpu-fp32 | 19.6 | 19.1 | 352.45 | 276.48 | 360.05 | 284.66 | 22.7 | 28.7 | +26% |
| onnxruntime-cpu-int8 | 16.0 | 15.8 | 437.26 | 363.21 | 441.92 | 365.72 | 18.3 | 22.0 | +20% |
| onnxruntime-cuda-fp32 | 37.9 | 38.2 | 152.74 | 73.3 | 178.18 | 100.37 | 51.9 | 101.5 | +96% |
| openvino-cpu-fp32 | 16.9 | 15.9 | 366.99 | 288.67 | 407.5 | 383.34 | 21.3 | 26.7 | +25% |
| openvino-cpu-int8 | 26.6 | 23.7 | 246.29 | 231.1 | 291.41 | 257.69 | 32.0 | 34.3 | +7% |
| tensorrt-cuda-fp16 | 51.3 | 48.3 | 108.88 | 53.38 | 117.83 | 65.38 | 72.5 | 145.4 | +101% |

Run-to-run noise on this laptop is about ±10% (single A/B rounds). The PyTorch rows are inside
it: two earlier A/B rounds of pytorch-cuda-fp32 at c=8 gave 38.9 → 35.3 and 39.0 → 35.2 req/s,
this one 40.3 → 43.7. For PyTorch the lock still covers its whole pipeline, because ultralytics
letterboxes and runs NMS inside its own predictor. The exported backends improved in every
round, and most where the model call is shortest relative to the CPU stages (TensorRT, ONNX
Runtime on CUDA). At c=1 there is nothing to overlap, and the differences are noise.
