# ─── Multi-stage build ──────────────────────────────────────
# Stage 1: Builder — install dependencies in a virtual env
# Stage 2: Runtime — copy only the venv + app code (smaller image)
# ────────────────────────────────────────────────────────────

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

# System libs required at runtime by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY .env.example .env

# Writable cache for model weights and the ultralytics settings file.
#
# WORKDIR is root-owned and the service runs as appuser, so anything that
# writes relative to the CWD fails with PermissionError. Ultralytics
# downloads a bare weight filename into the CWD and writes settings.json
# into its config dir, so both are pointed at appuser-owned paths under
# /home/appuser/.cache — the directory docker-compose mounts as a volume,
# which is what makes the weights survive a container restart.
ENV MODEL_WEIGHTS_DIR=/home/appuser/.cache/vision-inference/weights \
    YOLO_CONFIG_DIR=/home/appuser/.cache/ultralytics

RUN mkdir -p "$MODEL_WEIGHTS_DIR" "$YOLO_CONFIG_DIR" && \
    chown -R appuser:appuser /home/appuser/.cache

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
