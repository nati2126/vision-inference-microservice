"""Shared test fixtures.

The ``client`` fixture creates an ``httpx.AsyncClient`` backed by a real
FastAPI app (using ``ASGITransport``). This is the recommended approach
for testing FastAPI applications — it exercises the full middleware and
lifespan pipeline without starting a real server.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio as the async backend for pytest-asyncio."""
    return "asyncio"


@pytest.fixture(scope="session")
async def client():
    """Yield an async HTTP test client wired to the real app.

    ``scope="session"`` ensures the model is loaded only once across
    all tests — matching production behaviour and keeping the suite fast.
    """
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
