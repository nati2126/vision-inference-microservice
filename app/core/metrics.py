"""Prometheus metrics, served at GET /metrics."""

from prometheus_client import Counter, Gauge, Histogram

# Spans one fast GPU request (~20 ms) to a CPU server deep in its queue (seconds).
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

DETECT_REQUESTS = Counter(
    "vision_detect_requests_total",
    "POST /api/v1/detect responses, by HTTP status.",
    ["status"],
)
DETECT_DURATION = Histogram(
    "vision_detect_request_duration_seconds",
    "Time from accepting a detect request to its last response byte.",
    buckets=_LATENCY_BUCKETS,
)
IN_FLIGHT = Gauge(
    "vision_detect_in_flight_requests",
    "Detect requests accepted and not yet answered.",
)
REJECTED = Counter(
    "vision_detect_rejected_total",
    "Detect requests answered 503 because MAX_IN_FLIGHT_REQUESTS were already in flight.",
)
QUEUE_WAIT = Histogram(
    "vision_model_queue_wait_seconds",
    "Time a preprocessed request waited for its model call to start.",
    buckets=_LATENCY_BUCKETS,
)
MODEL_CALL = Histogram(
    "vision_model_call_duration_seconds",
    "Duration of one model call, which may serve a whole batch.",
    buckets=_LATENCY_BUCKETS,
)
BATCH_SIZE = Histogram(
    "vision_model_batch_size",
    "Requests served by each model call.",
    buckets=(1, 2, 3, 4, 6, 8, 12, 16, 32),
)
MODEL_INFO = Gauge(
    "vision_model_info",
    "The loaded model; always 1.",
    ["backend", "device", "precision", "max_batch_size"],
)
