import json

import httpx
import pytest
from sqlalchemy import select

from mathmodel_ai.agents import AgentRunStatus, ProblemAgent
from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import Environment
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import AgentRunRecord
from mathmodel_ai.db.session import (
    create_database_engine,
    create_session_factory,
    session_scope,
)
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.probe import ProviderCompatibilityProbe
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.secrets import EnvironmentSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import TaskProfile, TaskType
from mathmodel_ai.schemas.problem_analysis import ProblemAgentInput
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
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
from tests.reasoning.helpers import analysis_fixture


@pytest.mark.integration
@pytest.mark.asyncio
async def test_z_custom_provider_probe_router_problem_agent_and_audit_record() -> None:
    secret = "sk-custom-e2e-secret"
    settings = Settings(
        environment="test",
        database_url="sqlite+pysqlite:///:memory:",
        default_model_id="custom-main",
        default_provider="mock",
    )
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    secrets = EnvironmentSecretResolver({"CUSTOM_E2E_KEY": secret})
    security = EndpointSecurityPolicy(
        environment=Environment.TEST,
        resolver=lambda _host, _port: ["93.184.216.34"],
    )
    configurations = ProviderModelRegistry(
        session_factory,
        secrets=secrets,
        security_policy=security,
    )
    endpoint = configurations.create_provider(
        ProviderEndpoint(
            provider_id="custom-e2e",
            display_name="Custom E2E",
            protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
            base_url="https://custom-e2e.example.test/v1",
            credential_ref="env:CUSTOM_E2E_KEY",
            trust_level=EndpointTrustLevel.USER_MANAGED_PROXY,
        )
    )
    declared = {
        capability: CapabilityEvidence(
            status=CapabilityStatus.SUPPORTED,
            source=CapabilitySource.USER_DECLARED,
        )
        for capability in (
            ModelCapability.TEXT,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.JSON_SCHEMA,
            ModelCapability.SYSTEM_ROLE,
            ModelCapability.STREAMING,
            ModelCapability.USAGE_REPORTING,
        )
    }
    configurations.create_model(
        ModelProfile(
            model_id="custom-main",
            provider_id=endpoint.provider_id,
            display_name="Custom Main",
            remote_model="remote-custom-model",
            quality_tier=QualityTier.FLAGSHIP_MAX,
            declared_capabilities=declared,
            structured_output_strategy=StructuredOutputStrategy.NATIVE_JSON_SCHEMA,
        )
    )

    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.headers["authorization"] == f"Bearer {secret}"
        if payload.get("stream"):
            return httpx.Response(
                200,
                content=(b'data: {"choices":[{"delta":{"content":"O"}}]}\n\ndata: [DONE]\n\n'),
            )
        response_format = payload.get("response_format")
        content = "OK"
        if isinstance(response_format, dict):
            schema = response_format.get("json_schema", {})
            schema_name = schema.get("name") if isinstance(schema, dict) else None
            content = (
                analysis_fixture().model_dump_json()
                if schema_name == "ProblemAnalysis"
                else '{"ok":true,"value":1}'
            )
        return httpx.Response(
            200,
            json={
                "id": "custom-response",
                "model": "gpt-5.6-sol",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://custom-e2e.example.test/v1",
    )
    providers = ProviderRegistry(
        [],
        configurations=configurations,
        secrets=secrets,
        security_policy=security,
        configured_clients={endpoint.provider_id: client},
    )
    probe = await ProviderCompatibilityProbe(
        configurations=configurations,
        providers=providers,
    ).run("custom-main")
    assert probe.authentication_status is ProbeAuthenticationStatus.PASS
    assert probe.capabilities[ModelCapability.JSON_SCHEMA].status is CapabilityStatus.SUPPORTED

    router = ModelRouter(
        settings,
        available_providers=providers.available,
        registry=configurations,
    )
    agent = ProblemAgent(
        router=router,
        providers=providers,
        prompts=PromptRegistry(),
        max_retries=0,
    )
    repository = ReasoningRepository(session_factory)
    state = repository.create_project_problem(
        name="Custom Provider E2E",
        title="Demand allocation",
        raw_problem="Forecast demand, optimize allocation, and evaluate the resulting plan.",
    )
    run = await agent.run(
        ProblemAgentInput(title=state.title, raw_problem=state.raw_problem),
        state,
        TaskProfile(task_type=TaskType.PROBLEM_UNDERSTANDING),
    )
    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.output == analysis_fixture()
    assert run.is_mock is False
    assert run.provider_id == "custom-e2e"
    assert run.model_id == "custom-main"
    assert run.remote_model == "remote-custom-model"
    assert run.remote_model_reported == "gpt-5.6-sol"
    assert run.provider_config_digest == endpoint.config_digest
    assert run.protocol == ProviderProtocol.OPENAI_CHAT_COMPLETIONS.value
    assert run.endpoint_trust == EndpointTrustLevel.USER_MANAGED_PROXY.value
    assert run.structured_output_mode == StructuredOutputStrategy.NATIVE_JSON_SCHEMA.value
    repository.record_run(state.project_id, state.problem_id, run)

    with session_scope(session_factory) as session:
        row = session.scalar(select(AgentRunRecord).where(AgentRunRecord.id == run.run_id))
        assert row is not None
        assert row.remote_model == "remote-custom-model"
        assert row.model == "gpt-5.6-sol"
        trace = repr(
            {
                "provider_id": row.provider_id,
                "provider_config_digest": row.provider_config_digest,
                "model_id": row.model_id,
                "remote_model": row.remote_model,
                "model_config_digest": row.model_config_digest,
                "protocol": row.protocol,
                "structured_output_mode": row.structured_output_mode,
                "reasoning_requested": row.reasoning_requested,
                "reasoning_effective": row.reasoning_effective,
                "endpoint_trust": row.endpoint_trust,
                "is_mock": row.is_mock,
                "error": row.error,
                "route": row.route_json,
            }
        )
    assert secret not in trace
    assert all(payload["model"] == "remote-custom-model" for payload in requests)
    await providers.aclose()
    await client.aclose()
