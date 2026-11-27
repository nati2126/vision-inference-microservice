"""Load shedding and request metrics for the detect endpoint."""

import json
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core import metrics

_RETRY_AFTER_SECONDS = "1"


class AdmissionMiddleware:
    """Answers 503 once max_in_flight detect requests are being served.

    Runs before the upload is parsed or decoded, so a rejection costs little.
    Without a limit the queue grows without bound under overload and every
    request's latency grows with it.
    """

    def __init__(self, app: ASGIApp, path: str, max_in_flight: int) -> None:
        self.app = app
        self.path = path
        self.max_in_flight = max_in_flight
        self.in_flight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != self.path or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        if self.in_flight >= self.max_in_flight:
            metrics.REJECTED.inc()
            metrics.DETECT_REQUESTS.labels(status="503").inc()
            await _drain_body(receive)
            await _send_overloaded(send)
            return

        self.in_flight += 1
        metrics.IN_FLIGHT.set(self.in_flight)
        start = time.perf_counter()
        status = "500"  # if the app raises before starting a response

        async def send_and_record(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = str(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_and_record)
        finally:
            self.in_flight -= 1
            metrics.IN_FLIGHT.set(self.in_flight)
            metrics.DETECT_DURATION.observe(time.perf_counter() - start)
            metrics.DETECT_REQUESTS.labels(status=status).inc()


async def _drain_body(receive: Receive) -> None:
    """Read and discard the upload.

    Answering without reading it leaves the client blocked mid-upload: the
    server stops reading the socket, so the client never reads the 503.
    """
    while True:
        message = await receive()
        if message["type"] != "http.request" or not message.get("more_body", False):
            return


async def _send_overloaded(send: Send) -> None:
    body = json.dumps({"error": "Server is at capacity. Retry shortly.", "detail": None}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", _RETRY_AFTER_SECONDS.encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
