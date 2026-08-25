import pytest
from pydantic import ValidationError

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import ProviderName
from mathmodel_ai.providers.factory import build_provider_registry


def test_production_rejects_mock_default_provider() -> None:
    with pytest.raises(ValidationError, match="forbidden in production"):
        Settings(environment="production", default_provider="mock")


def test_production_requires_key_for_selected_provider() -> None:
    with pytest.raises(ValidationError, match="API key is required"):
        Settings(environment="production", default_provider="openai")


def test_secret_is_masked() -> None:
    settings = Settings(openai_api_key="secret-value")
    assert "secret-value" not in repr(settings)
    assert "secret-value" not in settings.model_dump_json()


def test_model_jury_weights_must_sum_to_one_hundred() -> None:
    with pytest.raises(ValidationError, match="weights must sum to 100"):
        Settings(model_jury_weights={"problem_fit": 24})


@pytest.mark.asyncio
async def test_production_registry_does_not_expose_mock_provider() -> None:
    settings = Settings(
        environment="production",
        default_provider="openai",
        openai_api_key="configured-secret",
    )
    registry = build_provider_registry(settings)
    try:
        assert registry.available == {ProviderName.OPENAI}
    finally:
        await registry.aclose()
