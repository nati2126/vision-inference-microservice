# TensorRT with pre/post-processing on the GPU

The laptop in `i9-14900hx_rtx-4070-laptop.md`, 2026-11-27, default batch-1 FP16
engine. "CPU" is the NumPy pipeline (commit `dbe9075`); "GPU" uploads the uint8 image
and filters detections on the GPU (commit `96c3b40`). They ran interleaved, CPU then
GPU, twice, so drift on the laptop hits both alike. The CPU was 6.8% busy with other
programs before the first round. Cells are round 1 / round 2.

## Model (`python -m benchmark.run_benchmark --only tensorrt-cuda-fp16 --skip-http`)

Single-image `predict()` on coco128:

| | CPU | GPU |
|---|---|---|
| mAP50 | 0.595 / 0.595 | 0.595 / 0.595 |
| mAP50-95 | 0.444 / 0.444 | 0.444 / 0.444 |
| p50 ms | 11.02 / 11.21 | 5.10 / 4.02 |
| p95 ms | 16.75 / 13.01 | 8.72 / 4.58 |
| img/s | 79.6 / 88.0 | 169.4 / 243.0 |
| Peak RSS MB | 806 / 806 | 910 / 878 |
| Peak GPU MB | 242 / 242 | 307 / 307 |

## HTTP (200 requests per level, one uvicorn worker, client on the same machine)

| | CPU req/s | GPU req/s | CPU p50 ms | GPU p50 ms |
|---|---|---|---|---|
| c=1 | 47.2 / 47.6 | 78.2 / 72.2 | 20.7 / 20.4 | 12.4 / 13.0 |
| c=8 | 141.6 / 132.8 | 175.2 / 180.9 | 55.3 / 58.9 | 44.2 / 43.2 |
| c=32 | 129.6 / 133.4 | 177.1 / 172.0 | 236.7 / 226.7 | 172.0 / 174.1 |

## Where the time went

Medians of 300 calls in one process, after warm-up, the GPU columns from commit
`96c3b40`. The two images are a 640x480 coco128 image and `bus.jpg` (810x1080):

| Stage | CPU 640x480 | GPU 640x480 | CPU 810x1080 | GPU 810x1080 |
|---|---:|---:|---:|---:|
| Preprocess | 4.80 | 0.51 | 6.02 | 2.05 |
| Model call, incl. transfers | 3.25 | 2.17 | 3.50 | 2.16 |
| Postprocess | 2.03 | 0.81 | 1.98 | 0.81 |

A first version that also ran NMS on the GPU (torchvision `batched_nms`) took
2.40–2.55 ms to postprocess. Most of that was the many small kernel launches and
host syncs, not the NMS itself.

- **Most of the CPU path's preprocessing was the float conversion**, not the resize:
  4.9 ms in total against 0.7 ms for `letterbox()`. The GPU path uploads 0.9 MB of
  uint8 instead of 4.9 MB of float32.
- **The resize stays in OpenCV.** `torch.nn.functional.interpolate` differed from
  `cv2.INTER_LINEAR` by one grey level on 8.6% of pixels. That moved an FP16 box on
  `bus.jpg` to IoU 0.973, below the 0.98 parity threshold.
- **Scaling uses a lookup table.** CUDA divides by a scalar as a multiply by its
  reciprocal, one ulp off NumPy for many of the 256 values. That cost 0.001 mAP50.
  With a NumPy-computed table, the GPU input is bit-identical to the CPU path's.
- **NMS stays in NumPy.** At the default confidence threshold of 0.25, about 50 of the
  8400 anchors survive the filter. NumPy NMS took 0.10 ms for them and
  `batched_nms` 0.52 ms. GPU NMS only won at a threshold of 0.001 (638 candidates:
  1.09 vs 5.19 ms), which the service does not use.
- **One stream, owned by the backend.** A fresh stream per call, from torch's pool of
  32, left the caching allocator holding 406 MB for 7 MB in use after 200 requests.
- **GPU memory is 65 MB higher** (307 vs 242 MB), and host RSS is 70–100 MB higher
  because torch's CUDA kernels are loaded.
- **The model-call time is bimodal** on this laptop, about 2.2 ms or about 5.3 ms,
  for the same code. It switches with the GPU's clock state between bursts. The CPU
  path shows the same (3.2–3.8 ms across runs). Within one process, copying the engine
  output instead of returning a view of it cost 0.1 ms (2.15 vs 2.07 ms).
