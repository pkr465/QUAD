"""E2E test configuration — use asyncio backend explicitly."""
import pytest


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param
