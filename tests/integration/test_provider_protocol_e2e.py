import json
from typing import Any

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
    CustomJSONConfiguration,
    CustomJSONRequestMapping,
    CustomJSONResponseMapping,
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


def _response_content(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True)
    if "preferred_interpretation" in serialized:
        return analysis_fixture().model_dump_json()
    if "Return only JSON matching" in serialized:
        return '{"ok":true,"value":1}'
    return "OK"


def _partial_handler(requests: list[dict[str, Any]], secret: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {secret}"
        payload = json.loads(request.content)
        requests.append(payload)
        if payload.get("stream"):
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"O"}}]}\n\ndata: [DONE]\n\n',
            )
        return httpx.Response(
            200,
            json={
                "id": "partial-response",
                "model": "vendor-name-claimed-by-proxy",
                "choices": [
                    {
                        "message": {"content": _response_content(payload)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )

    return httpx.MockTransport(handler)


def _custom_handler(requests: list[dict[str, Any]], secret: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-model-key"] == secret
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "result": {"text": _response_content(payload)},
                "usage": {"input": 4, "output": 2, "total": 6},
            },
        )

    return httpx.MockTransport(handler)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_kind", ["partial-compatible", "custom-json-http"])
async def test_provider_protocol_full_agent_e2e(adapter_kind: str) -> None:
    secret = "fake-e2e-secret"
    provider_id = f"{adapter_kind}-provider"
    model_id = f"{adapter_kind}-model"
    settings = Settings(
        environment="test",
        database_url="sqlite+pysqlite:///:memory:",
        default_model_id=model_id,
        default_provider="mock",
    )
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    secret_name = "PROTOCOL_E2E_KEY"
    secrets = EnvironmentSecretResolver({secret_name: secret})
    security = EndpointSecurityPolicy(
        environment=Environment.TEST,
        resolver=lambda _host, _port: ["93.184.216.34"],
    )
    configurations = ProviderModelRegistry(
        session_factory,
        secrets=secrets,
        security_policy=security,
    )
    protocol = (
        ProviderProtocol.OPENAI_CHAT_COMPLETIONS
        if adapter_kind == "partial-compatible"
        else ProviderProtocol.CUSTOM_JSON_HTTP
    )
    endpoint = configurations.create_provider(
        ProviderEndpoint(
            provider_id=provider_id,
            display_name=adapter_kind,
            protocol=protocol,
            base_url=f"https://{adapter_kind}.example.test/v1",
            credential_ref=f"env:{secret_name}",
            trust_level=EndpointTrustLevel.USER_MANAGED_PROXY,
        )
    )
    configuration: dict[str, Any] = {}
    if adapter_kind == "custom-json-http":
        configuration["custom_json"] = CustomJSONConfiguration(
            request=CustomJSONRequestMapping(
                endpoint_path="/generate",
                body={"model": "{model}", "prompt": "{prompt}", "limit": "{max_tokens}"},
            ),
            response=CustomJSONResponseMapping(
                text_path="result.text",
                input_tokens_path="usage.input",
                output_tokens_path="usage.output",
                total_tokens_path="usage.total",
            ),
            auth_scheme="API_KEY_HEADER",
            api_key_header="X-Model-Key",
        ).model_dump(mode="json")
    declared = {
        ModelCapability.TEXT: CapabilityEvidence(
            status=CapabilityStatus.SUPPORTED,
            source=CapabilitySource.USER_DECLARED,
        ),
        ModelCapability.STRUCTURED_OUTPUT: CapabilityEvidence(
            status=CapabilityStatus.SUPPORTED,
            source=CapabilitySource.USER_DECLARED,
        ),
    }
    configurations.create_model(
        ModelProfile(
            model_id=model_id,
            provider_id=provider_id,
            display_name=adapter_kind,
            remote_model=f"remote-{adapter_kind}",
            quality_tier=QualityTier.FLAGSHIP_MAX,
            declared_capabilities=declared,
            structured_output_strategy=StructuredOutputStrategy.PROMPT_JSON_FALLBACK,
            configuration=configuration,
        )
    )
    requests: list[dict[str, Any]] = []
    transport = (
        _partial_handler(requests, secret)
        if adapter_kind == "partial-compatible"
        else _custom_handler(requests, secret)
    )
    client = httpx.AsyncClient(
        transport=transport,
        base_url=f"https://{adapter_kind}.example.test/v1",
    )
    providers = ProviderRegistry(
        [],
        configurations=configurations,
        secrets=secrets,
        security_policy=security,
        configured_clients={provider_id: client},
    )

    probe = await ProviderCompatibilityProbe(
        configurations=configurations, providers=providers
    ).run(model_id)
    assert probe.authentication_status is ProbeAuthenticationStatus.PASS
    assert probe.capabilities[ModelCapability.STRUCTURED_OUTPUT].status is CapabilityStatus.PARTIAL
    router = ModelRouter(settings, registry=configurations, available_providers=set())
    agent = ProblemAgent(
        router=router,
        providers=providers,
        prompts=PromptRegistry(),
        max_retries=0,
    )
    repository = ReasoningRepository(session_factory)
    state = repository.create_project_problem(
        name=f"{adapter_kind} E2E",
        title="Allocation problem",
        raw_problem="Forecast demand and optimize a constrained allocation.",
    )
    profile = TaskProfile(task_type=TaskType.PROBLEM_UNDERSTANDING)
    run = await agent.run(
        ProblemAgentInput(title=state.title, raw_problem=state.raw_problem),
        state,
        profile,
    )

    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.output == analysis_fixture()
    assert run.is_mock is False
    assert run.provider_id == provider_id
    assert run.model_id == model_id
    assert run.remote_model == f"remote-{adapter_kind}"
    assert run.endpoint_trust == EndpointTrustLevel.USER_MANAGED_PROXY.value
    assert run.routes[-1].capability_probe_id == probe.probe_id
    assert run.routes[-1].task_profile_digest == profile.requirements_digest
    repository.record_run(state.project_id, state.problem_id, run)
    with session_scope(session_factory) as session:
        row = session.scalar(select(AgentRunRecord).where(AgentRunRecord.id == run.run_id))
        assert row is not None
        assert row.provider_id == provider_id
        assert row.model_id == model_id
        assert row.remote_model == f"remote-{adapter_kind}"
        assert row.protocol == protocol.value
        assert row.endpoint_trust == EndpointTrustLevel.USER_MANAGED_PROXY.value
    assert requests
    assert secret not in json.dumps(requests)
    assert endpoint.credential_ref == f"env:{secret_name}"
    await providers.aclose()
    await client.aclose()
