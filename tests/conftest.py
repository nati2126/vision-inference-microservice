"""Shared fixtures. Model artifacts are exported into models/ on first use."""

import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

# Must precede the ultralytics import.
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
    async with _app_client("pytorch", "yolov8n.pt") as ac:
        yield ac


@pytest.fixture(scope="session")
async def onnx_client(onnx_fp32_path: Path) -> AsyncGenerator[AsyncClient, None]:
    async with _app_client("onnxruntime", str(onnx_fp32_path)) as ac:
        yield ac


@pytest.fixture(scope="session")
def onnx_fp32_path() -> Path:
    from scripts.export_models import export_onnx_fp32

    MODELS_DIR.mkdir(exist_ok=True)
    return export_onnx_fp32(MODELS_DIR, force=False)


@pytest.fixture(scope="session")
def pt_path(onnx_fp32_path: Path) -> Path:
    return MODELS_DIR / "yolov8n.pt"


def _asset(name: str) -> npt.NDArray[np.uint8]:
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / name), cv2.IMREAD_COLOR)
    assert image is not None, f"missing ultralytics asset {name}"
    return np.asarray(image, dtype=np.uint8)


@pytest.fixture(scope="session")
def bus_image() -> npt.NDArray[np.uint8]:
    return _asset("bus.jpg")


@pytest.fixture(scope="session")
def zidane_image() -> npt.NDArray[np.uint8]:
    return _asset("zidane.jpg")
