from __future__ import annotations

import os
from uuid import uuid4

import pytest

from mathmodel_ai.agents.paper import PaperAgent
from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import Environment, ProviderName
from mathmodel_ai.paper.claims import ClaimEvidenceGraph, ClaimEvidenceValidator
from mathmodel_ai.providers.factory import build_provider_registry
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import EscalationLevel, TaskProfile, TaskType
from mathmodel_ai.schemas.paper import (
    ClaimEvidenceLink,
    ClaimEvidenceSupport,
    CompetitionProfile,
    PaperAgentInput,
)
from mathmodel_ai.schemas.problem_state import ProblemState
from tests.paper.helpers import evidence_snapshot, result_evidence


def _has_live_credential() -> bool:
    settings = Settings()
    return any(
        item is not None
        for item in (
            settings.openai_api_key,
            settings.google_api_key,
            settings.anthropic_api_key,
        )
    )


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("MM_RUN_LIVE_PROVIDER_TESTS") != "1",
        reason="SKIPPED_NO_CREDENTIALS: live PaperAgent test requires explicit opt in",
    ),
    pytest.mark.skipif(
        not _has_live_credential(),
        reason="SKIPPED_NO_CREDENTIALS: no live provider key is configured",
    ),
]


@pytest.mark.asyncio
async def test_live_paper_agent_emits_evidence_bound_paper_ir() -> None:
    base = Settings()
    assert base.default_provider is not ProviderName.MOCK
    catalog = base.model_catalog.model_copy(
        update={
            name: target.model_copy(
                update={
                    "provider": base.default_provider,
                    "model": base.default_provider_model,
                }
            )
            for name, target in base.model_catalog
        }
    )
    settings = base.model_copy(
        update={
            "environment": Environment.TEST,
            "model_catalog": catalog,
            "reasoning_max_retries": 1,
        }
    )
    providers = build_provider_registry(settings)
    evidence = result_evidence()
    paper_id = uuid4()
    state = ProblemState(
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        title="Live paper fixture",
        raw_problem="Explain a verified objective without introducing new facts.",
    )
    agent = PaperAgent(
        router=ModelRouter(settings, available_providers=providers.available),
        providers=providers,
        prompts=PromptRegistry(),
        max_retries=1,
    )
    try:
        run = await agent.run(
            PaperAgentInput(
                project_id=evidence.project_id,
                assigned_paper_id=paper_id,
                assigned_version=1,
                title=state.title,
                evidence=[evidence],
                evidence_snapshot=evidence_snapshot(evidence),
                competition_profile=CompetitionProfile(),
            ),
            state,
            TaskProfile(
                task_type=TaskType.PAPER_IR,
                complexity=4,
                reasoning_requirement=4,
                long_context_requirement=4,
                review_requirement=4,
                minimum_level=EscalationLevel.FLAGSHIP_HIGH,
            ),
        )
        assert run.output is not None, run.errors
        assert run.is_mock is False
        assert (run.output.paper_id, run.output.version) == (paper_id, 1)
        graph = ClaimEvidenceGraph([evidence])
        for claim in run.output.claims:
            graph.add_claim(claim)
            for evidence_id in claim.evidence_refs:
                graph.add_link(
                    ClaimEvidenceLink(
                        project_id=evidence.project_id,
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                        support=ClaimEvidenceSupport.DIRECT_SUPPORT,
                        source_field=(
                            claim.structured_value.source_field
                            if hasattr(claim.structured_value, "source_field")
                            else None
                        ),
                        rationale="live output explicitly bound this evidence",
                    )
                )
        assert ClaimEvidenceValidator().validate(graph) == []
    finally:
        await providers.aclose()
