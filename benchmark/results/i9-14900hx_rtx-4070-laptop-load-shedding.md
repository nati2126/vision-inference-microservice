# Load shedding under overload

`python -m benchmark.overload` (defaults), 2026-11-27, on the laptop in
`i9-14900hx_rtx-4070-laptop.md`. ONNX Runtime FP32 on the CPU (about 30 req/s at
capacity) with batching on, 256 clients in 4 processes for 45 s per run. Clients
honour `Retry-After` (1 s) after a 503. Two rounds per setting; round 1 / round 2.

| `MAX_IN_FLIGHT_REQUESTS` | Goodput req/s | Served p50 ms | Served p95 ms | Served p99 ms | Rejected (503) |
|---|---:|---:|---:|---:|---:|
| 64 (default) | 33.4 / 33.2 | 1854 / 1866 | 1925 / 1937 | 3132 / 2986 | 7690 / 7681 |
| 100000 (no limit) | 34.0 / 33.8 | 7198 / 7256 | 7251 / 7308 | 7276 / 7327 | 0 / 0 |

- **Effect:** the limit cuts the latency of served requests about 4x for 1-2% less goodput.
  That goodput is spent reading the rejected uploads. Without a limit, every request
  waits behind all the others: latency grows with the number of clients, here to about
  7 s.
- **Server-side check:** the service's own `vision_detect_request_duration_seconds`
  averaged 1.81 / 1.82 s with the limit and 6.49 / 6.54 s without it, matching the
  client.
- **How the limit is enforced:** rejections happen in ASGI middleware before the upload is
  parsed or decoded. The body is still read and discarded first: over a real socket an
  unread upload leaves the client blocked mid-send and the request hangs.
- **The load generator matters:** a first attempt with all 256 clients in one Python process
  saturated that process's core on 503 retries. It stopped feeding the server and made
  shedding look harmful (goodput 20 vs 31 req/s). That is why the tool spreads clients
  over processes.

## Overhead at normal load

Interleaved against the same code without the middleware and metrics: 200 requests per
level, three rounds, req/s at c=1 / 8 / 32.

| Config | Without | With |
|---|---|---|
| TensorRT FP16 | 42.2 / 118.3 / 121.5, 47.5 / 138.1 / 131.0, 47.8 / 138.0 / 133.3 | 42.7 / 125.5 / 114.6, 47.4 / 138.7 / 133.5, 47.1 / 135.0 / 134.6 |
| ONNX Runtime FP32 CPU | 18.5 / 35.2 / 35.4, 18.9 / 34.0 / 34.8, 19.0 / 34.3 / 34.9 | 18.6 / 33.7 / 35.0, 19.0 / 34.3 / 35.0, 18.8 / 34.5 / 35.1 |

No consistent difference; the largest single gap (6%) points different ways in different
rounds.
