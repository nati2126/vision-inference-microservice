# Benchmark results (full run)

- **Hardware:** Intel(R) Core(TM) i9-14900HX (24 cores / 32 threads), 34.1 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8.6 GB), Windows 11 (build 10.0.26200), Python 3.11.14, on AC power
- **Generated:** 2026-11-27T05:18:27+00:00, commit `d2765e8`, took 591s
- **Accuracy:** coco128 (first 128 images of COCO train2017), 128 images, conf 0.001, NMS IoU 0.7. A relative comparison between backends, not an official COCO result.
- **Latency:** `predict()` end to end (preprocess + inference + postprocess), batch 1, 10 warm-up + 100 timed runs on one 640x480 image, serving thresholds (conf 0.25).
- **Batching:** the model call alone (`infer_batch`, no pre/post-processing), 10 warm-up + 30 timed calls per batch size. PyTorch's call runs ultralytics' whole pipeline and only takes batch 1. Timed after the memory peaks were read, so the memory columns are for batch 1.
- **HTTP:** 200 `POST /api/v1/detect` requests per concurrency level against uvicorn (1 worker); client and server share the machine. Batch max is the server's `BATCH_MAX_SIZE`, capped by what the artifact accepts.

## Model

| Backend | Device | Precision | mAP50 | mAP50-95 | Mean ms | p50 ms | p95 ms | img/s | Size MB | Peak RSS MB | Peak GPU MB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 0.607 | 0.448 | 70.97 | 70.40 | 73.83 | 14.1 | 6.5 | 751 | n/a |
| pytorch | cuda | fp32 | 0.607 | 0.448 | 14.24 | 13.89 | 15.70 | 70.2 | 6.5 | 1378 | 357 |
| onnxruntime | cpu | fp32 | 0.594 | 0.444 | 42.39 | 42.35 | 44.84 | 23.6 | 12.7 | 172 | n/a |
| onnxruntime | cpu | int8 | 0.581 | 0.430 | 52.61 | 53.31 | 54.37 | 19.0 | 3.5 | 141 | n/a |
| onnxruntime | cuda | fp32 | 0.594 | 0.443 | 16.03 | 15.83 | 17.38 | 62.4 | 12.7 | 1155 | 221 |
| openvino | cpu | fp32 | 0.594 | 0.444 | 50.15 | 50.05 | 56.50 | 19.9 | 12.9 | 203 | n/a |
| openvino | cpu | int8 | 0.577 | 0.430 | 37.70 | 37.87 | 44.53 | 26.5 | 4.3 | 195 | n/a |
| tensorrt | cuda | fp16 | 0.596 | 0.443 | 14.18 | 12.88 | 18.91 | 70.5 | 9.6 | 837 | 594 |
| onnxruntime (ablation: naive INT8) | cpu | int8 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 3.5 | 128 | n/a |

## Batching (model call)

| Backend | Device | Precision | Max batch | batch 1 ms | batch 1 img/s | batch 8 ms | batch 8 img/s |
|---|---|---|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 1 | 71.96 | 13.9 | n/a | n/a |
| pytorch | cuda | fp32 | 1 | 14.22 | 70.3 | n/a | n/a |
| onnxruntime | cpu | fp32 | any | 34.00 | 29.4 | 208.52 | 38.4 |
| onnxruntime | cpu | int8 | any | 44.89 | 22.3 | 316.71 | 25.3 |
| onnxruntime | cuda | fp32 | 1 | 8.01 | 124.8 | n/a | n/a |
| openvino | cpu | fp32 | any | 33.89 | 29.5 | 238.06 | 33.6 |
| openvino | cpu | int8 | any | 31.16 | 32.1 | 204.93 | 39.0 |
| tensorrt | cuda | fp16 | 16 | 3.98 | 251.0 | 30.70 | 260.6 |

## HTTP

| Backend | Device | Precision | Batch max | c=1 p50 ms | c=1 p95 ms | c=1 req/s | c=8 p50 ms | c=8 p95 ms | c=8 req/s | c=32 p50 ms | c=32 p95 ms | c=32 req/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 1 | 80.5 | 86.7 | 12.3 | 576.4 | 602.1 | 13.7 | 2399.5 | 2513.3 | 13.1 |
| pytorch | cuda | fp32 | 1 | 27.6 | 44.2 | 32.5 | 174.1 | 251.9 | 42.6 | 652.3 | 727.2 | 48.5 |
| onnxruntime | cpu | fp32 | 1 | 52.3 | 54.5 | 19.1 | 282.8 | 291.5 | 28.2 | 1143.0 | 1157.7 | 27.8 |
| onnxruntime | cpu | fp32 | 8 | 52.5 | 54.3 | 19.1 | 232.2 | 272.7 | 34.1 | 885.9 | 909.9 | 35.5 |
| onnxruntime | cpu | int8 | 1 | 63.3 | 64.8 | 15.9 | 369.7 | 373.8 | 21.6 | 1479.3 | 1494.5 | 21.4 |
| onnxruntime | cpu | int8 | 8 | 63.4 | 65.4 | 15.8 | 335.3 | 365.4 | 23.7 | 1317.7 | 1345.0 | 24.0 |
| onnxruntime | cuda | fp32 | 1 | 24.9 | 33.0 | 38.6 | 88.3 | 106.2 | 87.9 | 392.6 | 440.0 | 79.0 |
| openvino | cpu | fp32 | 1 | 65.3 | 72.9 | 15.3 | 305.9 | 369.0 | 25.5 | 1168.5 | 1344.5 | 26.3 |
| openvino | cpu | fp32 | 8 | 64.7 | 71.4 | 15.4 | 322.0 | 389.9 | 24.1 | 1090.9 | 1195.2 | 28.3 |
| openvino | cpu | int8 | 1 | 47.5 | 60.3 | 20.9 | 239.2 | 278.3 | 33.2 | 977.6 | 1057.5 | 32.6 |
| openvino | cpu | int8 | 8 | 41.9 | 51.3 | 23.4 | 231.2 | 280.1 | 34.1 | 887.8 | 944.4 | 35.6 |
| tensorrt | cuda | fp16 | 1 | 21.2 | 26.6 | 45.0 | 68.3 | 83.6 | 113.8 | 294.9 | 351.8 | 102.1 |
| tensorrt | cuda | fp16 | 8 | 21.2 | 28.1 | 44.4 | 62.8 | 86.1 | 125.6 | 230.0 | 288.0 | 132.1 |
