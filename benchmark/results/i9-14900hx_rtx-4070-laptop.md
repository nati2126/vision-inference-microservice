# Benchmark results (full run)

- **Hardware:** Intel(R) Core(TM) i9-14900HX (24 cores / 32 threads), 34.1 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8.6 GB), Windows 11 (build 10.0.26200), Python 3.11.14, on AC power
- **Generated:** 2026-10-08T09:32:33+00:00, commit `1dd7e01`, took 378s
- **Accuracy:** coco128 (first 128 images of COCO train2017), 128 images, conf 0.001, NMS IoU 0.7. A relative comparison between backends, not an official COCO result.
- **Latency:** `predict()` end to end (preprocess + inference + postprocess), batch 1, 10 warm-up + 100 timed runs on one 640x480 image, serving thresholds (conf 0.25).
- **HTTP:** 200 `POST /api/v1/detect` requests per concurrency level against uvicorn (1 worker); client and server share the machine.

## Model

| Backend | Device | Precision | mAP50 | mAP50-95 | Mean ms | p50 ms | p95 ms | img/s | Size MB | Peak RSS MB | Peak GPU MB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 0.607 | 0.448 | 92.61 | 91.65 | 100.75 | 10.8 | 6.5 | 751 | n/a |
| pytorch | cuda | fp32 | 0.607 | 0.448 | 22.77 | 22.48 | 25.65 | 43.9 | 6.5 | 1379 | 357 |
| onnxruntime | cpu | fp32 | 0.594 | 0.444 | 40.44 | 40.60 | 42.27 | 24.7 | 12.8 | 172 | n/a |
| onnxruntime | cpu | int8 | 0.581 | 0.430 | 51.36 | 52.37 | 53.28 | 19.5 | 3.7 | 142 | n/a |
| onnxruntime | cuda | fp32 | 0.594 | 0.443 | 18.99 | 18.79 | 20.52 | 52.7 | 12.8 | 1152 | 221 |
| openvino | cpu | fp32 | 0.594 | 0.444 | 45.33 | 44.55 | 54.26 | 22.1 | 12.9 | 182 | n/a |
| openvino | cpu | int8 | 0.577 | 0.430 | 32.17 | 30.09 | 40.67 | 31.1 | 4.4 | 209 | n/a |
| tensorrt | cuda | fp16 | 0.595 | 0.444 | 13.84 | 13.92 | 15.00 | 72.2 | 9.4 | 807 | 242 |
| onnxruntime (ablation: naive INT8) | cpu | int8 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 3.6 | 128 | n/a |

## HTTP

| Backend | Device | Precision | c=1 p50 ms | c=1 p95 ms | c=1 req/s | c=8 p50 ms | c=8 p95 ms | c=8 req/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 115.3 | 125.5 | 8.6 | 766.9 | 817.0 | 10.4 |
| pytorch | cuda | fp32 | 30.9 | 33.5 | 32.0 | 196.5 | 257.5 | 39.0 |
| onnxruntime | cpu | fp32 | 50.3 | 52.0 | 19.9 | 346.5 | 356.4 | 23.1 |
| onnxruntime | cpu | int8 | 62.2 | 63.8 | 16.2 | 433.2 | 441.8 | 18.4 |
| onnxruntime | cuda | fp32 | 28.5 | 31.4 | 34.9 | 167.3 | 199.9 | 46.0 |
| openvino | cpu | fp32 | 60.5 | 69.8 | 16.6 | 401.6 | 445.8 | 19.7 |
| openvino | cpu | int8 | 45.0 | 54.6 | 21.8 | 314.5 | 362.1 | 25.2 |
| tensorrt | cuda | fp16 | 22.4 | 24.8 | 44.4 | 119.5 | 125.7 | 65.9 |
