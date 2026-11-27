# Benchmark results (full run)

- **Hardware:** Intel(R) Core(TM) i9-14900HX (24 cores / 32 threads), 34.1 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8.6 GB), Windows 11 (build 10.0.26200), Python 3.11.14, on AC power
- **Generated:** 2026-11-27T06:48:10+00:00, commit `c4f0111`, took 690s
- **Accuracy:** coco128 (first 128 images of COCO train2017), 128 images, conf 0.001, NMS IoU 0.7. A relative comparison between backends, not an official COCO result.
- **Latency:** `predict()` end to end (preprocess + inference + postprocess), batch 1, 10 warm-up + 100 timed runs on one 640x480 image, serving thresholds (conf 0.25).
- **Batching:** the model call alone (`infer_batch`, no pre/post-processing), 10 warm-up + 30 timed calls per batch size. PyTorch's call runs ultralytics' whole pipeline and only takes batch 1. Timed after the memory peaks were read, so the memory columns are for batch 1.
- **HTTP:** 200 `POST /api/v1/detect` requests per concurrency level against uvicorn (1 worker); client and server share the machine. Batch max is the server's `BATCH_MAX_SIZE`, capped by what the artifact accepts.

## Model

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

## Batching (model call)

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

## HTTP

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
