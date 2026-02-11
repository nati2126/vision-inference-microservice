# 🔍 Vision Inference Microservice

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

Production-grade object detection inference microservice powered by **YOLOv8** and **FastAPI**.

---

## Architecture

```
app/
├── api/v1/endpoints/     # Thin route handlers (API layer)
├── schemas/              # Pydantic request / response models
├── services/             # Business logic (service layer)
├── models/               # ML model wrappers (model layer)
└── core/                 # Config, logging, lifecycle
```

| Layer | Responsibility |
|-------|---------------|
| **API** | HTTP routing, input validation, error mapping |
| **Service** | Image decoding, inference orchestration, metadata |
| **Model** | YOLOv8 loading, raw inference, result parsing |
| **Core** | Configuration (12-factor), structured logging |

---

## Quick Start

### Local Development

```bash
# 1. Clone & enter
git clone https://github.com/your-user/vision-inference-microservice.git
cd vision-inference-microservice

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
make install  # or: pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env

# 5. Run development server
make dev  # or: uvicorn app.main:app --reload
```

### Docker

```bash
# Build & run with Docker Compose
make docker-up

# View logs
make docker-logs

# Stop
make docker-down
```

---

## API Reference

### `GET /api/v1/health`

Health check endpoint for load balancers and orchestrators.

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true
}
```

### `POST /api/v1/detect`

Upload an image and receive object detections.

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.9231,
      "bbox": {
        "x_min": 45.12,
        "y_min": 102.34,
        "x_max": 320.56,
        "y_max": 489.78
      }
    }
  ],
  "metadata": {
    "image_width": 640,
    "image_height": 480,
    "inference_time_ms": 42.15,
    "detections_count": 1
  }
}
```

### Interactive Docs

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Development

```bash
make lint        # Run ruff linter
make format      # Auto-format code
make typecheck   # Run mypy
make test        # Run pytest
make help        # Show all available targets
```

---

## Configuration

All settings are driven by environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `yolov8n.pt` | YOLOv8 model variant |
| `MODEL_CONFIDENCE_THRESHOLD` | `0.25` | Minimum detection confidence |
| `MODEL_DEVICE` | `cpu` | Compute device (`cpu` / `cuda` / `mps`) |
| `LOG_LEVEL` | `INFO` | Logging severity |
| `LOG_FORMAT` | `json` | Output format (`json` / `console`) |
| `ENVIRONMENT` | `development` | Runtime environment |

---

## Tech Stack

- **Python 3.10+** — Type-hinted, modern Python
- **FastAPI** — Async web framework with automatic OpenAPI docs
- **Ultralytics YOLOv8** — State-of-the-art object detection
- **PyTorch** — ML runtime
- **Pydantic v2** — Data validation and settings management
- **structlog** — Structured, machine-parsable logging
- **Docker** — Containerised deployment with multi-stage builds
- **pytest + httpx** — Async test suite

---

## License

MIT
