"""End-to-end HTTP benchmark: start the real service, load it with httpx.

This measures what a client sees: multipart upload, JPEG decode, inference,
JSON serialisation and the event loop, not just ``predict()``. The client
runs on the same machine as the server and competes with it for CPU, which
is noted alongside the results.
"""

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

_STARTUP_TIMEOUT_S = 180.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ServiceProcess:
    """Run ``uvicorn app.main:app`` for one backend in a child process."""

    def __init__(self, backend: str, model_path: str, device: str) -> None:
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._env = {
            **os.environ,
            "MODEL_BACKEND": backend,
            "MODEL_PATH": model_path,
            "MODEL_DEVICE": device,
            "LOG_LEVEL": "WARNING",
            "LOG_FORMAT": "json",
        }
        self._log = tempfile.NamedTemporaryFile(  # noqa: SIM115  # closed in stop()
            prefix="bench-server-", suffix=".log", delete=False
        )
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> dict[str, Any]:
        """Start the server and block until ``/health`` answers; return its body."""
        self._process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "app.main:app",
                "--host", "127.0.0.1", "--port", str(self.port),
                "--workers", "1", "--log-level", "warning",
            ],
            env=self._env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )  # fmt: skip
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"server exited during startup:\n{self.log_tail()}")
            try:
                response = httpx.get(f"{self.base_url}/api/v1/health", timeout=2.0)
                if response.status_code == 200:
                    body: dict[str, Any] = response.json()
                    return body
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"server not healthy after {_STARTUP_TIMEOUT_S}s:\n{self.log_tail()}")

    def log_tail(self, lines: int = 20) -> str:
        self._log.flush()
        text = Path(self._log.name).read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def stop(self) -> None:
        """Stop the server and wait until it has really exited.

        On Windows a venv's ``python.exe`` is a launcher that runs the real
        interpreter as a child; terminating the launcher ends that child a
        moment later. The child holds the log file open, so the log becoming
        deletable is the signal that the server is gone, and the next
        backend never starts while the previous one still holds CPU or GPU
        resources.
        """
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._log.close()

        deadline = time.monotonic() + 15
        while True:
            try:
                Path(self._log.name).unlink(missing_ok=True)
                return
            except PermissionError:
                if time.monotonic() > deadline:
                    print(f"warning: server log still locked, left at {self._log.name}")
                    return
                time.sleep(0.2)


async def _run_load(
    url: str, image: bytes, total: int, concurrency: int, warmup: int
) -> dict[str, Any]:
    """Send ``total`` requests from ``concurrency`` concurrent workers."""
    files_template = ("image.jpg", image, "image/jpeg")
    latencies_ms: list[float] = []
    errors = 0
    remaining = total

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
        for _ in range(warmup):
            await client.post(url, files={"file": files_template})

        async def worker() -> None:
            nonlocal remaining, errors
            while remaining > 0:
                remaining -= 1  # no await between check and decrement: race-free
                start = time.perf_counter()
                response = await client.post(url, files={"file": files_template})
                latencies_ms.append((time.perf_counter() - start) * 1000)
                if response.status_code != 200:
                    errors += 1

        wall_start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall_s = time.perf_counter() - wall_start

    lat = np.asarray(latencies_ms)
    return {
        "concurrency": concurrency,
        "requests": total,
        "errors": errors,
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "mean_ms": round(float(lat.mean()), 2),
        "requests_per_s": round(total / wall_s, 1),
    }


def run_http_benchmark(
    backend: str,
    model_path: str,
    device: str,
    image: bytes,
    total: int,
    concurrencies: tuple[int, ...],
    warmup: int,
) -> dict[str, Any]:
    """Start the service for one backend and load-test ``/detect``."""
    service = ServiceProcess(backend, model_path, device)
    try:
        health = service.start()
        url = f"{service.base_url}/api/v1/detect"
        runs = [
            asyncio.run(_run_load(url, image, total, concurrency, warmup))
            for concurrency in concurrencies
        ]
        return {"health": health, "runs": runs}
    finally:
        service.stop()
