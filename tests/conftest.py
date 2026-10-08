"""Shared test fixtures.

The client fixtures create an ``httpx.AsyncClient`` backed by a real
FastAPI app (using ``ASGITransport``).

``ASGITransport`` speaks HTTP to the app but does **not** emit the ASGI
lifespan events, so startup never runs and ``app.state`` stays empty. The
app is therefore entered through ``app.router.lifespan_context(app)``
explicitly, which loads the model and populates ``app.state`` exactly as it
would under uvicorn.

Model artifacts live in ``models/`` (gitignored). The PyTorch weights and
the ONNX FP32 export are produced on first use, so a fresh checkout or CI
runner needs no separate export step; INT8, OpenVINO and TensorRT
artifacts come from ``python -m scripts.export_models`` and the tests that
need them skip when they are absent.
"""

import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

# Before anything imports ultralytics: never let it pip-install packages
# (e.g. the CPU onnxruntime over an installed onnxruntime-gpu).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@asynccontextmanager
async def _app_client(backend: str, model_path: str) -> AsyncIterator[AsyncClient]:
    """Start the real app on ``backend`` and yield a client for it."""
    # Pin the configuration so a developer's .env or shell cannot change
    # what the tests exercise. Environment variables take precedence over
    # .env, and the lifespan reads the settings cached here.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MODEL_BACKEND", backend)
        patch.setenv("MODEL_PATH", model_path)
        patch.setenv("MODEL_DEVICE", "cpu")
        get_settings.cache_clear()
        app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac
    get_settings.cache_clear()


@pytest.fixture(scope="session")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Client for the default (PyTorch) app.

    ``scope="session"`` ensures the model is loaded only once across
    all tests — matching production behaviour and keeping the suite fast.
    """
    async with _app_client("pytorch", "yolov8n.pt") as ac:
        yield ac


@pytest.fixture(scope="session")
async def onnx_client(onnx_fp32_path: Path) -> AsyncGenerator[AsyncClient, None]:
    """Client for the app served by ONNX Runtime on CPU."""
    async with _app_client("onnxruntime", str(onnx_fp32_path)) as ac:
        yield ac


@pytest.fixture(scope="session")
def onnx_fp32_path() -> Path:
    """``models/yolov8n.onnx``, exported on first use."""
    from scripts.export_models import export_onnx_fp32

    MODELS_DIR.mkdir(exist_ok=True)
    return export_onnx_fp32(MODELS_DIR, force=False)


@pytest.fixture(scope="session")
def pt_path(onnx_fp32_path: Path) -> Path:
    """The ``.pt`` weights the ONNX file was exported from."""
    return MODELS_DIR / "yolov8n.pt"


def _asset(name: str) -> npt.NDArray[np.uint8]:
    # ultralytics ships these two sample photos inside the package.
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / name), cv2.IMREAD_COLOR)
    assert image is not None, f"missing ultralytics asset {name}"
    return np.asarray(image, dtype=np.uint8)


@pytest.fixture(scope="session")
def bus_image() -> npt.NDArray[np.uint8]:
    """810x1080 street photo: a bus and four people."""
    return _asset("bus.jpg")


@pytest.fixture(scope="session")
def zidane_image() -> npt.NDArray[np.uint8]:
    """1280x720 photo: two people and a tie."""
    return _asset("zidane.jpg")
