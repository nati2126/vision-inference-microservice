# COCO val2017 accuracy

- **Hardware:** Intel(R) Core(TM) i9-14900HX (24 cores / 32 threads), 34.1 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8.6 GB), Windows 11 (build 10.0.26200), Python 3.11.14, on AC power, CPU 8.5% busy with other programs before the run
- **Generated:** 2026-11-27T10:30:55+00:00, commit `50f9ec4`, took 3706s
- **Data:** COCO val2017, 5000 images, conf 0.001, NMS IoU 0.7, max 300 detections, one class per box. Exported models take a 640x640 letterbox; PyTorch takes ultralytics' minimal rectangle (640x480 for a 4:3 image).
- **Scoring:** pycocotools (`COCOeval`, bbox), which keeps the top 100 detections per image and ignores detections on crowd regions. The in-house column is this repository's `DetectionEvaluator` on the same detections: all 300, non-crowd ground truth only, so detections on crowds count as false positives.
- **Images/s** is the evaluation loop (disk read, decode, `predict()` at conf 0.001), batch 1; it is not the latency benchmark.

| Backend | Device | Precision | mAP50-95 | mAP50 | mAP75 | small | medium | large | In-house mAP50-95 | In-house mAP50 | Images/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | cpu | fp32 | 0.367 | 0.517 | 0.399 | 0.178 | 0.405 | 0.520 | 0.363 | 0.511 | 8.7 |
| pytorch | cuda | fp32 | 0.367 | 0.517 | 0.399 | 0.179 | 0.406 | 0.520 | 0.363 | 0.511 | 29.5 |
| onnxruntime | cpu | fp32 | 0.367 | 0.516 | 0.399 | 0.178 | 0.405 | 0.519 | 0.363 | 0.510 | 16.0 |
| onnxruntime | cpu | int8 | 0.356 | 0.503 | 0.387 | 0.170 | 0.395 | 0.499 | 0.352 | 0.497 | 13.2 |
| onnxruntime | cuda | fp32 | 0.367 | 0.517 | 0.399 | 0.178 | 0.405 | 0.519 | 0.363 | 0.511 | 18.7 |
| openvino | cpu | fp32 | 0.367 | 0.516 | 0.399 | 0.178 | 0.405 | 0.519 | 0.363 | 0.510 | 11.2 |
| openvino | cpu | int8 | 0.356 | 0.503 | 0.386 | 0.169 | 0.394 | 0.500 | 0.352 | 0.497 | 17.1 |
| tensorrt | cuda | fp16 | 0.367 | 0.517 | 0.399 | 0.177 | 0.405 | 0.519 | 0.363 | 0.511 | 30.5 |
| onnxruntime (ablation: naive INT8) | cpu | int8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 16.2 |
