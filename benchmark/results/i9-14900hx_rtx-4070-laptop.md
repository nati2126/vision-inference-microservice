# Benchmark results (full run)

- **Hardware:** Intel(R) Core(TM) i9-14900HX (24 cores / 32 threads), 34.1 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8.6 GB), Windows 11 (build 10.0.26200), Python 3.11.14, on AC power, CPU 6.5% busy with other programs before the run
- **Generated:** 2026-11-27T08:25:42+00:00, commit `96c3b40`, took 545s
- **Accuracy:** coco128 (first 128 images of COCO train2017), 128 images, conf 0.001, NMS IoU 0.7. A relative comparison between backends, not an official COCO result.
- **Latency:** `predict()` end to end (preprocess + inference + postprocess), batch 1, 10 warm-up + 100 timed runs on one 640x480 image, serving thresholds (conf 0.25).
- **Batching:** the model call alone (`infer_batch`, no pre/post-processing), 10 warm-up + 30 timed calls per batch size. PyTorch's call runs ultralytics' whole pipeline and only takes batch 1. Timed after the memory peaks were read, so the memory columns are for batch 1.
- **HTTP:** 200 `POST /api/v1/detect` requests per concurrency level against uvicorn (1 worker); client and server share the machine. Batch max is the server's `BATCH_MAX_SIZE`, capped by what the artifact accepts.

## Model

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

## Batching (model call)

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

## HTTP

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
