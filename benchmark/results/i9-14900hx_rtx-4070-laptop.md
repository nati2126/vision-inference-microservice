# Benchmark results (full run)

- **Hardware:** Intel(R) Core(TM) i9-14900HX (24 cores / 32 threads), 34.1 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8.6 GB), Windows 10 (10.0.26200), Python 3.11.14, on AC power
- **Generated:** 2026-10-08T09:14:54+00:00, commit `da3835c`, took 388s
- **Accuracy:** coco128 (first 128 images of COCO train2017), 128 images, conf 0.001, NMS IoU 0.7. A relative comparison between backends, not an official COCO result.
- **Latency:** `predict()` end to end (preprocess + inference + postprocess), batch 1, 10 warm-up + 100 timed runs on one 640x480 image, serving thresholds (conf 0.25).
- **HTTP:** 200 `POST /api/v1/detect` requests per concurrency level against uvicorn (1 worker); client and server share the machine.

## Model

| Backend | Device | Precision | mAP50 | mAP50-95 | Mean ms | p50 ms | p95 ms | img/s | Size MB | Peak RSS MB | Peak GPU MB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 0.607 | 0.448 | 98.07 | 96.73 | 110.37 | 10.2 | 6.5 | 753 | n/a |
| pytorch | cuda | fp32 | 0.607 | 0.448 | 22.75 | 22.42 | 24.39 | 44.0 | 6.5 | 1382 | 357 |
| onnxruntime | cpu | fp32 | 0.594 | 0.444 | 42.77 | 42.48 | 45.17 | 23.4 | 12.8 | 170 | n/a |
| onnxruntime | cpu | int8 | 0.581 | 0.430 | 52.31 | 52.69 | 53.88 | 19.1 | 3.7 | 142 | n/a |
| onnxruntime | cuda | fp32 | 0.594 | 0.443 | 19.78 | 19.59 | 21.47 | 50.6 | 12.8 | 1153 | 221 |
| openvino | cpu | fp32 | 0.594 | 0.444 | 46.89 | 46.79 | 52.70 | 21.3 | 12.9 | 186 | n/a |
| openvino | cpu | int8 | 0.577 | 0.430 | 33.22 | 31.97 | 40.90 | 30.1 | 4.4 | 207 | n/a |
| tensorrt | cuda | fp16 | 0.595 | 0.444 | 13.25 | 13.23 | 13.85 | 75.5 | 9.4 | 805 | 242 |
| onnxruntime (ablation: naive INT8) | cpu | int8 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 3.6 | 127 | n/a |

## HTTP

| Backend | Device | Precision | c=1 p50 ms | c=1 p95 ms | c=1 req/s | c=8 p50 ms | c=8 p95 ms | c=8 req/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 124.3 | 143.1 | 8.0 | 826.4 | 880.5 | 9.6 |
| pytorch | cuda | fp32 | 31.8 | 35.4 | 31.1 | 203.3 | 266.1 | 37.6 |
| onnxruntime | cpu | fp32 | 50.9 | 53.5 | 19.5 | 349.0 | 357.6 | 22.9 |
| onnxruntime | cpu | int8 | 63.0 | 65.6 | 15.8 | 437.6 | 449.7 | 18.2 |
| onnxruntime | cuda | fp32 | 28.8 | 32.4 | 34.2 | 174.3 | 209.0 | 43.6 |
| openvino | cpu | fp32 | 59.2 | 70.9 | 16.6 | 377.4 | 461.9 | 20.7 |
| openvino | cpu | int8 | 46.9 | 56.0 | 20.9 | 313.4 | 345.1 | 25.3 |
| tensorrt | cuda | fp16 | 22.4 | 24.9 | 44.2 | 116.9 | 121.9 | 67.3 |
