"""Load shedding on /api/v1/detect, and the Prometheus endpoint."""

import asyncio
import io
from collections.abc import Callable

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY
from starlette.types import Receive, Scope, Send

from app.core.admission import AdmissionMiddleware

DETECT = "/api/v1/detect"


class _HeldApp:
    """Answers 200 once released; counts the requests that reached it."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.calls += 1
        await self.release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


async def _until(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    for _ in range(int(timeout_s / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


@pytest.mark.asyncio(loop_scope="session")
async def test_requests_over_the_limit_get_503_with_retry_after() -> None:
    inner = _HeldApp()
    middleware = AdmissionMiddleware(inner, path=DETECT, max_in_flight=1)
    rejected_before = _sample("vision_detect_rejected_total")

    async with AsyncClient(transport=ASGITransport(app=middleware), base_url="http://t") as c:
        first = asyncio.create_task(c.post(DETECT))
        await _until(lambda: middleware.in_flight == 1)

        overflow = await c.post(DETECT)
        inner.release.set()
        held = await first
        after = await c.post(DETECT)

    assert overflow.status_code == 503
    assert overflow.headers["retry-after"] == "1"
    assert overflow.json() == {"error": "Server is at capacity. Retry shortly.", "detail": None}
    assert inner.calls == 2, "the rejected request must not reach the app"
    assert held.status_code == 200
    assert after.status_code == 200
    assert middleware.in_flight == 0
    assert _sample("vision_detect_rejected_total") == rejected_before + 1


@pytest.mark.asyncio(loop_scope="session")
async def test_other_routes_are_not_limited() -> None:
    inner = _HeldApp()
    inner.release.set()
    middleware = AdmissionMiddleware(inner, path=DETECT, max_in_flight=1)
    middleware.in_flight = 1  # as if a detect request were being served

    async with AsyncClient(transport=ASGITransport(app=middleware), base_url="http://t") as c:
        health = await c.get("/api/v1/health")
        get_detect = await c.get(DETECT)

    assert health.status_code == 200
    assert get_detect.status_code == 200


@pytest.mark.asyncio(loop_scope="session")
async def test_metrics_endpoint_reports_detect_requests_and_the_model(
    client: AsyncClient,
) -> None:
    ok_before = _sample("vision_detect_requests_total", {"status": "200"})
    calls_before = _sample("vision_model_call_duration_seconds_count")
    _, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))

    detect = await client.post(
        DETECT, files={"file": ("t.jpg", io.BytesIO(jpeg.tobytes()), "image/jpeg")}
    )
    response = await client.get("/metrics")

    assert detect.status_code == 200
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert _sample("vision_detect_requests_total", {"status": "200"}) == ok_before + 1
    assert _sample("vision_model_call_duration_seconds_count") == calls_before + 1
    assert 'vision_model_info{backend="pytorch"' in response.text


@pytest.mark.asyncio(loop_scope="session")
async def test_a_rejected_upload_is_read_before_the_503() -> None:
    # Over real sockets an unread upload leaves the client blocked mid-send,
    # never reading the 503 (in-process transports cannot show this).
    middleware = AdmissionMiddleware(_HeldApp(), path=DETECT, max_in_flight=1)
    middleware.in_flight = 1
    chunks = [
        {"type": "http.request", "body": b"a", "more_body": True},
        {"type": "http.request", "body": b"b", "more_body": True},
        {"type": "http.request", "body": b"c", "more_body": False},
    ]
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return chunks.pop(0)

    async def send(message: dict[str, object]) -> None:
        assert not chunks, "responded before the whole upload was read"
        sent.append(message)

    scope = {"type": "http", "path": DETECT, "method": "POST"}
    await middleware(scope, receive, send)  # type: ignore[arg-type]

    assert sent[0]["status"] == 503
