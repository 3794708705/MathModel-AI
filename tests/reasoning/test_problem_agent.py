from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.agents import AgentRunStatus, ProblemAgent
from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import ProviderName
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.reasoning.quality_gates import understand_quality_gate
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import TaskProfile, TaskType
from mathmodel_ai.schemas.problem_analysis import (
    AmbiguityReviewStatus,
    EvidenceStatus,
    EvidenceType,
    ProblemAgentInput,
    SubProblemDependency,
)
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.quality import QualityGateStatus
from tests.reasoning.helpers import analysis_fixture


@pytest.mark.asyncio
async def test_problem_agent_preserves_ambiguity_and_proposed_assumption() -> None:
    response = analysis_fixture(low_confidence_ambiguity=True)
    mock = MockProvider([response.model_dump_json()])
    providers = ProviderRegistry([mock])
    agent = ProblemAgent(
        router=ModelRouter(Settings(), available_providers={ProviderName.MOCK}),
        providers=providers,
        prompts=PromptRegistry(),
        max_retries=0,
        ambiguity_review_threshold=0.7,
    )
    state = ProblemState(
        project_id=uuid4(),
        title="Demand allocation",
        raw_problem="Forecast demand, optimize allocation, and evaluate the resulting plan.",
    )
    result = await agent.run(
        ProblemAgentInput(title=state.title, raw_problem=state.raw_problem),
        state,
        TaskProfile(task_type=TaskType.PROBLEM_UNDERSTANDING),
    )

    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.output is not None
    assert result.output.human_review_recommended is True
    assert (
        result.output.ambiguities[0].review_status is AmbiguityReviewStatus.HUMAN_REVIEW_RECOMMENDED
    )
    assumption = result.output.assumptions_required[0]
    assert assumption.type is EvidenceType.ASSUMPTION
    assert assumption.status is EvidenceStatus.PROPOSED
    assert result.prompt_version == "2.1.0"
    assert result.is_mock is True
    gate = understand_quality_gate(result.output)
    assert gate.status is QualityGateStatus.PASS
    assert gate.warnings


def test_problem_agent_retry_receives_exact_schema_feedback() -> None:
    agent = ProblemAgent(router=None, providers=None, prompts=PromptRegistry())
    original = ProblemAgentInput(
        title="Match flow", raw_problem="Analyze point-by-point tennis match flow."
    )
    repaired = agent.prepare_attempt_input(
        original, None, ("subproblem Q2 dependencies contain unknown ids ['Q9']",)
    )
    assert "unknown ids ['Q9']" in repaired.repair_feedback[0]
    assert original.repair_feedback == []


def test_problem_analysis_reports_unknown_dependency_identity() -> None:
    payload = analysis_fixture().model_dump()
    payload["subproblems"][1]["input_dependencies"].append("Q9")
    with pytest.raises(ValidationError, match=r"subproblem Q2 dependencies.*Q9"):
        type(analysis_fixture()).model_validate(payload)


def test_problem_analysis_covers_prediction_optimization_and_evaluation_chain() -> None:
    analysis = analysis_fixture()
    assert [item.subproblem_id for item in analysis.subproblems] == ["Q1", "Q2", "Q3"]
    assert {
        (edge.upstream_id, edge.downstream_id) for edge in analysis.dependencies_between_subproblems
    } == {
        ("Q1", "Q2"),
        ("Q2", "Q3"),
    }
    assert analysis.assumptions_required[0].status is EvidenceStatus.PROPOSED


def test_problem_analysis_rejects_cyclic_subproblem_dependencies() -> None:
    payload = analysis_fixture().model_dump()
    payload["dependencies_between_subproblems"].append(
        SubProblemDependency(
            upstream_id="Q3",
            downstream_id="Q1",
            transferred_output="cyclic feedback",
        ).model_dump()
    )
    with pytest.raises(ValidationError, match="must be acyclic"):
        type(analysis_fixture()).model_validate(payload)
