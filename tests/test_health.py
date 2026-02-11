"""Tests for the health-check endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    """GET /api/v1/health should return 200 with expected fields."""
    response = await client.get("/api/v1/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert "version" in body
    assert body["model_loaded"] is True


@pytest.mark.asyncio
async def test_health_response_shape(client: AsyncClient) -> None:
    """Verify the response body matches the HealthResponse schema."""
    response = await client.get("/api/v1/health")
    body = response.json()

    required_keys = {"status", "version", "model_loaded"}
    assert required_keys.issubset(body.keys())
