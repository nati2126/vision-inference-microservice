"""Shared test fixtures.

The ``client`` fixture creates an ``httpx.AsyncClient`` backed by a real
FastAPI app (using ``ASGITransport``).

``ASGITransport`` speaks HTTP to the app but does **not** emit the ASGI
lifespan events, so startup never runs and ``app.state`` stays empty. The
app is therefore entered through ``app.router.lifespan_context(app)``
explicitly, which loads the model and populates ``app.state`` exactly as it
would under uvicorn.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app


@pytest.fixture(scope="session")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Yield an async HTTP test client wired to the real app.

    ``scope="session"`` ensures the model is loaded only once across
    all tests — matching production behaviour and keeping the suite fast.
    """
    # Pin the backend so a developer's .env or shell cannot change what the
    # API tests exercise. Environment variables take precedence over .env.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MODEL_BACKEND", "pytorch")
        patch.setenv("MODEL_PATH", "yolov8n.pt")
        patch.setenv("MODEL_DEVICE", "cpu")
        get_settings.cache_clear()
        app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac
