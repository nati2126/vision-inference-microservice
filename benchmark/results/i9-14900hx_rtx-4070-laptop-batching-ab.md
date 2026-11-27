# Dynamic batching: does it help?

200 `POST /api/v1/detect` requests per concurrency level, uvicorn with one worker, httpx
client on the same machine (the laptop in `i9-14900hx_rtx-4070-laptop.md`), 2026-11-27.

## Against main, interleaved

main served one request at a time behind a lock, with fixed-batch artifacts. This branch
serves through the batching worker with dynamic-batch artifacts. The configs were run in
turn, twice (round 1 / round 2), so slow drift on the laptop hits all of them alike. Both
branch rows are at commit `d2765e8`.

| Config | Server | c=1 req/s | c=8 req/s | c=32 req/s |
|---|---|---:|---:|---:|
| tensorrt-cuda-fp16 | main (fixed-batch engine, inference lock) | 39.5 / 42.1 | 120.3 / 122.6 | 119.1 / 117.4 |
| tensorrt-cuda-fp16 | this branch, `BATCH_MAX_SIZE=1` | 38.2 / 39.2 | 103.6 / 103.5 | 101.6 / 100.4 |
| tensorrt-cuda-fp16 | this branch, `BATCH_MAX_SIZE=8` | 39.5 / 48.5 | 114.8 / 140.6 | 121.5 / 164.8 |
| onnxruntime-cpu-fp32 | main (fixed-batch model, inference lock) | 20.2 / 20.1 | 30.3 / 30.2 | 30.0 / 30.5 |
| onnxruntime-cpu-fp32 | this branch, `BATCH_MAX_SIZE=1` | 19.8 / 19.9 | 29.3 / 29.8 | 29.0 / 29.5 |
| onnxruntime-cpu-fp32 | this branch, `BATCH_MAX_SIZE=8` | 19.6 / 19.8 | 35.7 / 35.1 | 36.8 / 37.0 |

- **ONNX Runtime on the CPU gains 16-23% at concurrency 8 and 32**, and both rounds agree.
- **TensorRT does not clearly gain.** With batching off, the dynamic engine and the worker
  together cost 14-16% against main. With batching on, round 1 is 5% below main at
  concurrency 8 and round 2 is 15% above, so the spread is wider than the effect. At about
  120-150 req/s the single server process is limited by its CPU work (decode, letterbox,
  NMS, HTTP), not by the 4-6 ms model call, so few requests are waiting when a call starts.

## Batching off vs on, in the published full run

From `i9-14900hx_rtx-4070-laptop.json` (commit `d2765e8`): the same server code with
`BATCH_MAX_SIZE=1` (off) and `=8` (on). Only artifacts that accept batches are listed;
PyTorch and ONNX Runtime on CUDA serve batch 1 either way.

| Config | c=1 req/s off / on | c=8 req/s off / on | c=8 change | c=32 req/s off / on | c=32 change | c=32 p95 ms off / on |
|---|---:|---:|---:|---:|---:|---:|
| onnxruntime-cpu-fp32 | 19.1 / 19.1 | 28.2 / 34.1 | +21% | 27.8 / 35.5 | +28% | 1157.7 / 909.9 |
| onnxruntime-cpu-int8 | 15.9 / 15.8 | 21.6 / 23.7 | +10% | 21.4 / 24.0 | +12% | 1494.5 / 1345.0 |
| openvino-cpu-fp32 | 15.3 / 15.4 | 25.5 / 24.1 | -5% | 26.3 / 28.3 | +8% | 1344.5 / 1195.2 |
| openvino-cpu-int8 | 20.9 / 23.4 | 33.2 / 34.1 | +3% | 32.6 / 35.6 | +9% | 1057.5 / 944.4 |
| tensorrt-cuda-fp16 | 45.0 / 44.4 | 113.8 / 125.6 | +10% | 102.1 / 132.1 | +29% | 351.8 / 288.0 |

One earlier full run at the same commit is not published. Its TensorRT rows collapsed
(49.8 req/s at concurrency 8 with batching on), and the two standalone rounds above, run
straight afterwards, did not reproduce that. At concurrency 1 there is nothing to batch,
and the differences there are noise.
