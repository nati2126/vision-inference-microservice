# Benchmarks

**Hardware:** Intel Core i9-14900HX (24 cores: 8P + 16E, 32 threads), 32 GB RAM,
NVIDIA GeForce RTX 4070 Laptop GPU (8 GB, driver 591.86), Windows 11, on AC power.
Python 3.11, torch 2.6.0+cu124, ultralytics 8.3.57, onnxruntime-gpu 1.20.2,
OpenVINO 2026.4.1, TensorRT 10.16.1. yolov8n, 640x640 input.
Full output, including every package version and the commit:
[`benchmark/results/i9-14900hx_rtx-4070-laptop.md`](../benchmark/results/i9-14900hx_rtx-4070-laptop.md)
(and `.json`).

## Speed and accuracy (coco128)

Accuracy on **coco128** (128 images, 929 boxes), confidence 0.001, NMS IoU 0.7,
identical evaluation code for every backend. coco128 is a small, quick set
for comparing the backends with each other; the full COCO val2017 set, scored
by pycocotools, is [below](#accuracy-on-coco-val2017). Latency is `predict()` end to end (preprocess + inference
+ postprocess), batch 1, 10 warm-up + 100 timed runs on a 640x480 image.

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

The ONNX and OpenVINO artifacts take a dynamic batch. At batch 1 that costs a
little: run back to back against the earlier fixed-batch exports, `predict()`
was 0–9% slower (ONNX Runtime CPU 43.4 → 43.4–44.3 ms p50). The TensorRT
engine takes one image per call and is built from a static export, which keeps
it level with the earlier engine (see [design](design.md)).

These numbers move between runs on this laptop. Before this run the CPU was
6.5% busy with other programs; in the run before it, 18.5% (mostly a web
browser), and every CPU row was slower (ONNX Runtime FP32: 48.9 vs 40.9 ms
p50). Compare rows within one table; the interleaved A/Bs linked below are the
better evidence for any single change.

## Accuracy on COCO val2017

All 5,000 val2017 images, scored by pycocotools (`COCOeval`), the reference
COCO evaluator. Same settings as above: confidence 0.001, NMS IoU 0.7, up to
300 detections and one class per box. Full output:
[`benchmark/results/i9-14900hx_rtx-4070-laptop-coco-val2017.md`](../benchmark/results/i9-14900hx_rtx-4070-laptop-coco-val2017.md)
(and `.json`).

| Backend | Device | Precision | mAP50-95 | mAP50 | mAP75 | small | medium | large |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 0.367 | 0.517 | 0.399 | 0.178 | 0.405 | 0.520 |
| pytorch | cuda | fp32 | 0.367 | 0.517 | 0.399 | 0.179 | 0.406 | 0.520 |
| onnxruntime | cpu | fp32 | 0.367 | 0.516 | 0.399 | 0.178 | 0.405 | 0.519 |
| onnxruntime | cpu | int8 | 0.356 | 0.503 | 0.387 | 0.170 | 0.395 | 0.499 |
| onnxruntime | cuda | fp32 | 0.367 | 0.517 | 0.399 | 0.178 | 0.405 | 0.519 |
| openvino | cpu | fp32 | 0.367 | 0.516 | 0.399 | 0.178 | 0.405 | 0.519 |
| openvino | cpu | int8 | 0.356 | 0.503 | 0.386 | 0.169 | 0.394 | 0.500 |
| tensorrt | cuda | fp16 | 0.367 | 0.517 | 0.399 | 0.177 | 0.405 | 0.519 |
| onnxruntime (ablation: naive INT8) | cpu | int8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

- **Every FP32 and FP16 backend lands within 0.0007** of the others
  (mAP50-95 0.3668–0.3675), TensorRT FP16 included.
- **INT8 costs 0.011 mAP50-95** in both runtimes (0.356 vs 0.367). The
  calibration images come from coco128, a slice of train2017, so this set is
  independent of them.
- **Ultralytics publishes 37.3 for yolov8n; this service scores 36.7.** Its
  validator differs in two ways: NMS may keep a box under several classes
  (`multi_label`), and images are batched into shared rectangles. I have not
  measured how much each contributes.
- **The in-house evaluator reads 0.004 lower** (0.363). It scores all 300
  detections, and with crowd boxes left out of the ground truth, detections on
  crowds count as false positives; pycocotools keeps 100 per image and ignores
  crowd regions. On synthetic scenes without either difference the two agree
  to 1e-6 (`tests/test_metrics.py`). The coco128 numbers above use the
  in-house evaluator.

## Batching

The model call alone (`infer_batch`, no pre/post-processing) at batch 1 and 8.
ONNX Runtime on CUDA and the default TensorRT engine take batch 1 (see [design](design.md)), and PyTorch's call is ultralytics' whole pipeline, which the
service runs one image at a time.

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

200 `POST /api/v1/detect` requests (the same 640x480 JPEG) per concurrency
level, uvicorn with one worker, httpx async client on the same machine.
`Batch max` is the server's `BATCH_MAX_SIZE` (1 = batching off), capped by
what the artifact accepts. The change per backend, and an interleaved
comparison against the previous serving code:
[`benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md`](../benchmark/results/i9-14900hx_rtx-4070-laptop-batching-ab.md).
The step before that, narrowing the inference lock to the model call:
[`benchmark/results/i9-14900hx_rtx-4070-laptop-serving-ab.md`](../benchmark/results/i9-14900hx_rtx-4070-laptop-serving-ab.md).

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

## Reproduce

```bash
pip install -e ".[openvino,tensorrt,bench]"   # whichever extras fit your machine
python -m scripts.export_models                 # first TensorRT build: ~8 min
python -m scripts.export_models --trt-max-batch 16   # a TensorRT engine that batches
python -m benchmark.run_benchmark               # full: ~11 min here
python -m benchmark.run_benchmark --quick       # smoke run: ~3 min here
python -m benchmark.overload                    # load shedding under overload: ~4 min
python -m benchmark.coco_val                    # COCO val2017 (downloads ~1 GB): ~62 min here
python -m benchmark.coco_val --limit 200        # its first 200 images, for a quick check
```

Backends whose package, artifact or GPU is missing are listed as skipped. Each
backend's model measurements run in a fresh subprocess so peak memory is its
own. Results are written to `benchmark/results/<hardware-tag>.{json,md}`.
