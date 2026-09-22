from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mathmodel_ai.benchmark.repository import BenchmarkRepository
from mathmodel_ai.core.types import Environment
from mathmodel_ai.db.base import Base
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.secrets import EnvironmentSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.benchmark import (
    BenchmarkConfig,
    BenchmarkPricing,
    BenchmarkRun,
    BenchmarkRunStatus,
    benchmark_run_digest,
)
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilitySource,
    CapabilityStatus,
    EndpointTrustLevel,
    ModelCapability,
    ModelProfile,
    ProviderEndpoint,
    ProviderProtocol,
    QualityTier,
)

ZERO = "0" * 64


def test_x_disabling_model_does_not_rewrite_historical_benchmark_identity() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    registry = ProviderModelRegistry(
        factory,
        secrets=EnvironmentSecretResolver({"HISTORY_KEY": "configured"}),
        security_policy=EndpointSecurityPolicy(
            environment=Environment.TEST,
            resolver=lambda _host, _port: ["93.184.216.34"],
        ),
    )
    endpoint = registry.create_provider(
        ProviderEndpoint(
            provider_id="history-provider",
            display_name="History Provider",
            protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
            base_url="https://history.example.test/v1",
            credential_ref="env:HISTORY_KEY",
            trust_level=EndpointTrustLevel.USER_MANAGED_PROXY,
        )
    )
    model = registry.create_model(
        ModelProfile(
            model_id="history-model",
            provider_id=endpoint.provider_id,
            display_name="History Model",
            remote_model="remote-history-model",
            quality_tier=QualityTier.FLAGSHIP_HIGH,
            declared_capabilities={
                ModelCapability.TEXT: CapabilityEvidence(
                    status=CapabilityStatus.SUPPORTED,
                    source=CapabilitySource.USER_DECLARED,
                )
            },
        )
    )
    run = BenchmarkRun(
        code_commit="a" * 40,
        source_tree_digest="b" * 64,
        working_tree_dirty=True,
        config=BenchmarkConfig(
            provider="custom",
            model=model.remote_model,
            provider_id=endpoint.provider_id,
            provider_config_digest=endpoint.config_digest,
            model_id=model.model_id,
            model_config_digest=model.config_digest,
            protocol=endpoint.protocol.value,
            endpoint_trust=endpoint.trust_level.value,
            model_identity_confidence="NOT_INDEPENDENTLY_VERIFIED",
            reasoning_tier="high",
            pricing=BenchmarkPricing(version="history-test"),
        ),
        case_manifest_digests={"BENCH-history": "c" * 64},
        status=BenchmarkRunStatus.RUNNING,
        run_digest=ZERO,
    )
    run = run.model_copy(update={"run_digest": benchmark_run_digest(run)})
    benchmarks = BenchmarkRepository(factory)
    benchmarks.create_run(run)

    disabled = registry.set_model_enabled(model.model_id, False)
    historical = benchmarks.get_run(run.run_id)
    assert disabled.config_digest != model.config_digest
    assert historical.config.model_id == model.model_id
    assert historical.config.model_config_digest == model.config_digest
    assert historical.config.provider_config_digest == endpoint.config_digest
    assert historical.run_digest == run.run_digest
