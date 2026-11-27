"""Overload the service and compare MAX_IN_FLIGHT_REQUESTS settings.

Clients run in several processes, because one Python process handling a few
hundred clients and their 503 retries saturates its own core and stops
feeding the server. Clients honour Retry-After. Server-side numbers come from
the service's own /metrics.

    python -m benchmark.overload --backend onnxruntime --model models/yolov8n.onnx \
        --limits 64 100000 --processes 4 --clients-per-process 64
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from benchmark.coco128 import ensure_coco128, image_paths
from benchmark.http_load import ServiceProcess


def _client_process(url: str, image: bytes, clients: int, duration_s: float, out: Any) -> None:
    async def run() -> dict[str, Any]:
        ok_ms: list[float] = []
        rejected = errors = 0
        stop = time.monotonic() + duration_s
        limits = httpx.Limits(max_connections=clients, max_keepalive_connections=clients)
        async with httpx.AsyncClient(timeout=300.0, limits=limits) as client:

            async def worker() -> None:
                nonlocal rejected, errors
                while time.monotonic() < stop:
                    start = time.perf_counter()
                    try:
                        response = await client.post(
                            url, files={"file": ("image.jpg", image, "image/jpeg")}
                        )
                    except httpx.TransportError:
                        errors += 1
                        continue
                    if response.status_code == 200:
                        ok_ms.append((time.perf_counter() - start) * 1000)
                    elif response.status_code == 503:
                        rejected += 1
                        await asyncio.sleep(float(response.headers.get("retry-after", "1")))
                    else:
                        errors += 1

            await asyncio.gather(*(worker() for _ in range(clients)))
        return {"ok_ms": ok_ms, "rejected": rejected, "errors": errors}

    out.put(asyncio.run(run()))


def _metrics(base_url: str) -> dict[str, float]:
    samples = {}
    for line in httpx.get(f"{base_url}/metrics").text.splitlines():
        if line.startswith("vision_"):
            name, value = line.rsplit(" ", 1)
            samples[name] = float(value)
    return samples


def run(args: argparse.Namespace, limit: int, image: bytes) -> dict[str, Any]:
    service = ServiceProcess(
        args.backend, args.model, args.device, {"MAX_IN_FLIGHT_REQUESTS": str(limit)}
    )
    try:
        service.start()
        url = f"{service.base_url}/api/v1/detect"
        before = _metrics(service.base_url)
        queue: Any = mp.Queue()
        procs = [
            mp.Process(
                target=_client_process,
                args=(url, image, args.clients_per_process, args.duration, queue),
            )
            for _ in range(args.processes)
        ]
        start = time.perf_counter()
        for proc in procs:
            proc.start()
        results = [queue.get() for _ in procs]
        for proc in procs:
            proc.join()
        wall_s = time.perf_counter() - start
        after = _metrics(service.base_url)
    finally:
        service.stop()

    def delta(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)

    ok = np.concatenate([np.asarray(r["ok_ms"]) for r in results])
    served = max(1.0, delta("vision_detect_request_duration_seconds_count"))
    return {
        "max_in_flight": limit,
        "clients": args.processes * args.clients_per_process,
        "served": int(ok.size),
        "rejected": sum(r["rejected"] for r in results),
        "errors": sum(r["errors"] for r in results),
        "goodput_req_s": round(ok.size / wall_s, 1),
        "p50_ms": round(float(np.percentile(ok, 50))),
        "p95_ms": round(float(np.percentile(ok, 95))),
        "p99_ms": round(float(np.percentile(ok, 99))),
        # Served requests only: rejections are answered before the timer starts.
        "server_mean_s": round(delta("vision_detect_request_duration_seconds_sum") / served, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument("--model", default="models/yolov8n.onnx")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limits", type=int, nargs="+", default=[64, 100_000])
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--clients-per-process", type=int, default=64)
    parser.add_argument("--duration", type=float, default=45.0, help="seconds per run")
    args = parser.parse_args()

    image = Path(image_paths(ensure_coco128(), 1)[0]).read_bytes()
    for limit in args.limits:
        print(json.dumps(run(args, limit, image)), flush=True)


if __name__ == "__main__":
    main()
