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
    settings = Settings(
        openai_api_key="secret-value",
        secret_master_key="c2VjcmV0LW1hc3Rlci1rZXktMzItYnl0ZXMtbG9uZyEhISE=",
    )
    assert "secret-value" not in repr(settings)
    assert "secret-value" not in settings.model_dump_json()
    assert "secret-master" not in repr(settings)


def test_cors_requires_explicit_origins() -> None:
    with pytest.raises(ValidationError, match="explicit HTTP"):
        Settings(cors_origins=["*"])
    with pytest.raises(ValidationError, match="explicit HTTP"):
        Settings(cors_origins=["https://example.test/path"])


def test_default_cors_supports_the_reserved_vite_development_ports() -> None:
    assert Settings().cors_origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]


def test_model_jury_weights_must_sum_to_one_hundred() -> None:
    with pytest.raises(ValidationError, match="weights must sum to 100"):
        Settings(model_jury_weights={"problem_fit": 24})


def test_automatic_model_repair_never_exceeds_three_iterations() -> None:
    with pytest.raises(ValidationError):
        Settings(phase5_max_repair_cycles=4)


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
