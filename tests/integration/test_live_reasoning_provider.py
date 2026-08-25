import os

import pytest

from mathmodel_ai.core.config import Settings
from mathmodel_ai.providers.factory import build_provider_registry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("MM_RUN_LIVE_PROVIDER_TESTS") != "1",
        reason="paid live-provider tests require explicit MM_RUN_LIVE_PROVIDER_TESTS=1",
    ),
]


@pytest.mark.asyncio
async def test_live_provider_registry_is_explicitly_configured() -> None:
    settings = Settings()
    registry = build_provider_registry(settings)
    try:
        assert settings.default_provider in registry.available
        assert settings.default_provider.value != "mock"
    finally:
        await registry.aclose()
