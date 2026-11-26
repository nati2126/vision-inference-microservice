"""/health, /livez and /readyz."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.mark.asyncio(loop_scope="session")
async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert "version" in body
    assert body["model_loaded"] is True


@pytest.mark.asyncio(loop_scope="session")
async def test_health_response_shape(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    body = response.json()

    required_keys = {"status", "version", "model_loaded", "backend", "precision", "device"}
    assert required_keys.issubset(body.keys())


@pytest.mark.asyncio(loop_scope="session")
async def test_health_reports_active_backend(client: AsyncClient) -> None:
    body = (await client.get("/api/v1/health")).json()

    assert body["backend"] == "pytorch"
    assert body["precision"] == "fp32"
    assert body["device"] == "cpu"


@pytest.mark.asyncio(loop_scope="session")
async def test_health_reports_onnxruntime_backend(onnx_client: AsyncClient) -> None:
    body = (await onnx_client.get("/api/v1/health")).json()

    assert body["status"] == "healthy"
    assert body["backend"] == "onnxruntime"
    assert body["precision"] == "fp32"


@pytest.mark.asyncio(loop_scope="session")
async def test_livez_and_readyz_pass_once_the_model_is_loaded(client: AsyncClient) -> None:
    live = await client.get("/livez")
    ready = await client.get("/readyz")

    assert (live.status_code, live.json()) == (200, {"status": "ok"})
    assert (ready.status_code, ready.json()) == (200, {"status": "ready"})


class _UnloadedModel:
    is_loaded = False


@pytest.mark.asyncio(loop_scope="session")
async def test_readyz_is_503_while_the_model_is_not_loaded() -> None:
    app = create_app()
    app.state.model = _UnloadedModel()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        live = await ac.get("/livez")
        ready = await ac.get("/readyz")

    assert live.status_code == 200
    assert (ready.status_code, ready.json()) == (503, {"status": "not_ready"})
