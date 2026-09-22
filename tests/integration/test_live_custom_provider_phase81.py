import json
import os

import pytest

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import Environment
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.session import create_database_engine, create_session_factory
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.probe import ProviderCompatibilityProbe
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.secrets import EnvironmentSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilityProbeResult,
    CapabilitySource,
    CapabilityStatus,
    EndpointTrustLevel,
    ModelCapability,
    ModelProfile,
    ProbeAuthenticationStatus,
    ProviderEndpoint,
    ProviderProtocol,
    QualityTier,
    StructuredOutputStrategy,
)

pytestmark = pytest.mark.integration


def _declared_live_capabilities() -> dict[ModelCapability, CapabilityEvidence]:
    return {
        capability: CapabilityEvidence(
            status=CapabilityStatus.SUPPORTED,
            source=CapabilitySource.USER_DECLARED,
        )
        for capability in (
            ModelCapability.TEXT,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.JSON_MODE,
            ModelCapability.REASONING_CONTROL,
            ModelCapability.SYSTEM_ROLE,
            ModelCapability.STREAMING,
            ModelCapability.USAGE_REPORTING,
        )
    }


async def _run_live_probe(
    endpoint: ProviderEndpoint,
    model: ModelProfile,
) -> CapabilityProbeResult:
    settings = Settings(
        environment="test",
        database_url="sqlite+pysqlite:///:memory:",
        default_provider="mock",
    )
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    secrets = EnvironmentSecretResolver()
    security = EndpointSecurityPolicy(environment=Environment.TEST)
    configurations = ProviderModelRegistry(
        create_session_factory(engine),
        secrets=secrets,
        security_policy=security,
    )
    configurations.create_provider(endpoint)
    configurations.create_model(model)
    providers = ProviderRegistry(
        [],
        configurations=configurations,
        secrets=secrets,
        security_policy=security,
    )
    try:
        return await ProviderCompatibilityProbe(
            configurations=configurations,
            providers=providers,
        ).run(model.model_id)
    finally:
        await providers.aclose()
        engine.dispose()


def _assert_live_matrix(result: CapabilityProbeResult) -> None:
    assert result.authentication_status is ProbeAuthenticationStatus.PASS
    assert result.capabilities[ModelCapability.TEXT].status is CapabilityStatus.SUPPORTED
    assert result.capabilities[ModelCapability.STRUCTURED_OUTPUT].status in {
        CapabilityStatus.SUPPORTED,
        CapabilityStatus.PARTIAL,
    }
    assert result.capabilities[ModelCapability.REASONING_CONTROL].status in {
        CapabilityStatus.SUPPORTED,
        CapabilityStatus.UNSUPPORTED,
    }
    assert result.capabilities[ModelCapability.USAGE_REPORTING].status is CapabilityStatus.SUPPORTED


@pytest.mark.asyncio
async def test_live_deepseek_compatible_probe() -> None:
    remote_model = os.getenv("MM_TEST_DEEPSEEK_MODEL_ID")
    if not remote_model or not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("SKIPPED_NO_CREDENTIALS: DeepSeek model/key not configured")
    endpoint = ProviderEndpoint(
        provider_id="deepseek-live-test",
        display_name="DeepSeek Live Test",
        protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
        base_url="https://api.deepseek.com",
        credential_ref="env:DEEPSEEK_API_KEY",
        trust_level=EndpointTrustLevel.OFFICIAL_VENDOR,
    )
    model = ModelProfile(
        model_id="deepseek-live-model",
        provider_id=endpoint.provider_id,
        display_name="DeepSeek Live Model",
        remote_model=remote_model,
        quality_tier=QualityTier.FLAGSHIP_HIGH,
        declared_capabilities=_declared_live_capabilities(),
        structured_output_strategy=StructuredOutputStrategy.JSON_MODE,
        reasoning_mapping={"LOW": "low"},
    )
    _assert_live_matrix(await _run_live_probe(endpoint, model))


@pytest.mark.asyncio
async def test_live_qwen_compatible_probe() -> None:
    remote_model = os.getenv("MM_TEST_QWEN_MODEL_ID")
    base_url = os.getenv("QWEN_BASE_URL")
    if not remote_model or not base_url or not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("SKIPPED_NO_CREDENTIALS: Qwen endpoint/model/key not configured")
    endpoint = ProviderEndpoint(
        provider_id="qwen-live-test",
        display_name="Qwen Live Test",
        protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
        base_url=base_url,
        credential_ref="env:DASHSCOPE_API_KEY",
        trust_level=EndpointTrustLevel.USER_MANAGED_PROXY,
    )
    model = ModelProfile(
        model_id="qwen-live-model",
        provider_id=endpoint.provider_id,
        display_name="Qwen Live Model",
        remote_model=remote_model,
        quality_tier=QualityTier.FLAGSHIP_HIGH,
        declared_capabilities=_declared_live_capabilities(),
        structured_output_strategy=StructuredOutputStrategy.JSON_MODE,
        reasoning_mapping={"LOW": "low"},
    )
    _assert_live_matrix(await _run_live_probe(endpoint, model))


@pytest.mark.asyncio
async def test_live_generic_custom_provider_probe() -> None:
    raw = os.getenv("MM_TEST_CUSTOM_PROVIDER_CONFIG")
    if not raw:
        pytest.skip("SKIPPED_NO_CREDENTIALS: custom provider config not configured")
    payload = json.loads(raw)
    endpoint = ProviderEndpoint.model_validate(payload["provider"])
    model = ModelProfile.model_validate(payload["model"])
    if not EnvironmentSecretResolver().is_configured(endpoint.credential_ref):
        pytest.skip("SKIPPED_NO_CREDENTIALS: custom credential_ref is not configured")
    result = await _run_live_probe(endpoint, model)
    assert result.authentication_status is ProbeAuthenticationStatus.PASS
    assert result.capabilities[ModelCapability.TEXT].status is CapabilityStatus.SUPPORTED
