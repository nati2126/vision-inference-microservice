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

from app.main import create_app


@pytest.fixture(scope="session")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Yield an async HTTP test client wired to the real app.

    ``scope="session"`` ensures the model is loaded only once across
    all tests — matching production behaviour and keeping the suite fast.
    """
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac
