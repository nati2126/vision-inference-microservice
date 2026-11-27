# Dynamic batching: does it help?

200 `POST /api/v1/detect` requests per concurrency level, uvicorn with one worker, httpx
client on the same machine (the laptop in `i9-14900hx_rtx-4070-laptop.md`), 2026-11-27.
In the interleaved A/Bs the servers ran in turn, round after round, so drift on the
laptop hits all of them alike; cells are round 1 / round 2 (/ round 3). Absolute numbers
differ between tables because the machine's background load did.

## Where batching is on: ONNX Runtime and OpenVINO on the CPU

ONNX Runtime FP32 on the CPU against main (commit `c596d9c`; the batching path has not
changed since):

| Server | c=1 req/s | c=8 req/s | c=32 req/s |
|---|---:|---:|---:|
| main (fixed-batch model, lock) | 20.2 / 20.1 | 30.3 / 30.2 | 30.0 / 30.5 |
| batching worker, `BATCH_MAX_SIZE=1` | 19.8 / 19.9 | 29.3 / 29.8 | 29.0 / 29.5 |
| batching worker, `BATCH_MAX_SIZE=8` | 19.6 / 19.8 | 35.7 / 35.1 | 36.8 / 37.0 |

Batching serves 16-23% more than main at concurrency 8 and 32, in both rounds.

Batching off vs on in the published full run (`i9-14900hx_rtx-4070-laptop.json`, commit
`7001169`), same server code with `BATCH_MAX_SIZE=1` and `=8`:

| Config | c=1 req/s off / on | c=8 req/s off / on | c=8 change | c=32 req/s off / on | c=32 change | c=32 p95 ms off / on |
|---|---:|---:|---:|---:|---:|---:|
| onnxruntime-cpu-fp32 | 17.9 / 18.0 | 26.1 / 27.8 | +7% | 25.6 / 27.7 | +8% | 1367.9 / 1230.1 |
| onnxruntime-cpu-int8 | 14.7 / 14.8 | 19.9 / 22.0 | +11% | 19.9 / 22.0 | +11% | 1680.6 / 1470.0 |
| openvino-cpu-fp32 | 14.7 / 14.4 | 21.4 / 25.0 | +17% | 20.9 / 26.0 | +24% | 1593.8 / 1320.3 |
| openvino-cpu-int8 | 19.9 / 19.4 | 29.4 / 29.9 | +2% | 29.3 / 33.3 | +14% | 1160.2 / 1034.5 |

That run's gains are smaller than in the A/B above and in the run before it at `c596d9c`
(ONNX Runtime FP32 +21% / +28%, OpenVINO INT8 +3% / +9%). Right after it, the CPU was 18.5%
busy with other programs (mostly a web browser), and CPU-bound serving is where that
shows. OpenVINO results vary the most between runs: it already spreads one image across
all cores. A run before that one at `7001169` was discarded: its ONNX Runtime row without
batching was 20.7 req/s at concurrency 32 against 26.5-28.1 in five earlier runs, which
would have overstated the gain.

## Where batching is off: TensorRT

The default TensorRT engine takes batch 1 and is built from a static export, and at a
batch limit of 1 the service serialises with a lock instead of the worker. Against main
(commit `7001169`):

| Server | c=1 req/s | c=8 req/s | c=32 req/s |
|---|---:|---:|---:|
| main (fixed-batch engine, lock) | 31.9 / 38.7 / 37.5 | 79.1 / 103.2 / 98.5 | 101.4 / 105.2 / 96.7 |
| this branch (batch-1 engine from a static export, lock) | 38.9 / 36.3 / 38.0 | 98.6 / 100.0 / 99.1 | 101.6 / 110.7 / 94.2 |

Level with main.

### Why: the batch-16 engine (`--trt-max-batch 16`, commit `c596d9c`)

| Server | c=1 req/s | c=8 req/s | c=32 req/s |
|---|---:|---:|---:|
| main (fixed-batch engine, lock) | 39.5 / 42.1 | 120.3 / 122.6 | 119.1 / 117.4 |
| batch-16 engine, worker, `BATCH_MAX_SIZE=1` | 38.2 / 39.2 | 103.6 / 103.5 | 101.6 / 100.4 |
| batch-16 engine, worker, `BATCH_MAX_SIZE=8` | 39.5 / 48.5 | 114.8 / 140.6 | 121.5 / 164.8 |

- With batching off, the batch-16 engine and the worker cost 14-16% against main.
- With batching on, round 1 is 5% below main at concurrency 8 and round 2 15% above: the
  spread is wider than the effect.
- The single server process is limited by its CPU work (decode, letterbox, NMS, HTTP) at
  this rate, not by the 4-6 ms model call, so few requests wait when a call starts.
- An engine built from the dynamic-batch ONNX is ~5% slower at batch 1 even with the batch
  fixed (15.0-15.2 vs 14.0-14.5 ms `predict()` p50), and the batch-16 profile reserves
  594 MB of GPU memory instead of 242 MB.
