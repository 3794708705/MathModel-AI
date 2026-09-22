from uuid import uuid4

import pytest

from mathmodel_ai.agents.base import AgentRunStatus
from mathmodel_ai.agents.explorer import ModelExplorer
from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.errors import ProviderResponseError
from mathmodel_ai.core.types import ProviderName
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.providers.schemas import GenerationRequest, ModelResponse
from mathmodel_ai.reasoning.deduplication import deduplicate_candidates
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.reasoning.quality_gates import explore_quality_gate
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import TaskProfile, TaskType
from mathmodel_ai.schemas.model_selection import ModelExploration, ModelExplorerInput, ModelFamily
from mathmodel_ai.schemas.problem_analysis import DataAvailability
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.quality import QualityGateStatus
from tests.reasoning.helpers import analysis_fixture, candidate_fixture, exploration_fixture


def test_model_chain_candidate_covers_prediction_optimization_and_evaluation() -> None:
    exploration = exploration_fixture()
    required = {"Q1", "Q2", "Q3"}
    gate = explore_quality_gate(exploration, required)
    assert gate.status is QualityGateStatus.PASS
    assert exploration.candidates[0].family is ModelFamily.MODEL_CHAIN
    assert set(exploration.candidates[0].target_subproblems) == required


def test_semantic_deduplication_handles_milp_aliases() -> None:
    variants = [
        candidate_fixture("CAND-a", name="MILP", family=ModelFamily.MILP, targets=["Q2"]),
        candidate_fixture(
            "CAND-b",
            name="Mixed Integer Linear Programming",
            family=ModelFamily.MILP,
            targets=["Q2"],
        ),
        candidate_fixture(
            "CAND-c", name="整数线性规划模型", family=ModelFamily.MILP, targets=["Q2"]
        ),
    ]
    result = deduplicate_candidates(variants)
    assert [item.candidate_id for item in result.candidates] == ["CAND-a"]
    assert result.removed_ids == ["CAND-b", "CAND-c"]


def test_explore_gate_rejects_duplicate_candidates() -> None:
    first = candidate_fixture("CAND-one", name="MILP", family=ModelFamily.MILP, targets=["Q2"])
    duplicate = candidate_fixture(
        "CAND-two",
        name="Mixed Integer Linear Programming",
        family=ModelFamily.MILP,
        targets=["Q2"],
    )
    exploration = ModelExploration(
        candidates=[first, duplicate],
        fewer_than_three_reason="Only one model family is represented in this test.",
        exploration_summary="Deliberately duplicated candidate set.",
    )
    gate = explore_quality_gate(exploration, {"Q2"})
    assert gate.status is QualityGateStatus.RETRY
    assert "semantically_distinct" in gate.errors


@pytest.mark.asyncio
async def test_explorer_retry_receives_error_without_mutating_original_input() -> None:
    requests: list[GenerationRequest] = []
    diagnostic = "json_syntax=Invalid escape; line=3; column=12; position=42"

    class RecordingProvider(MockProvider):
        async def generate(self, request: GenerationRequest) -> ModelResponse:
            requests.append(request)
            response = await super().generate(request)
            if len(requests) == 1:
                raise ProviderResponseError(diagnostic)
            return response

    provider = RecordingProvider(["malformed", exploration_fixture().model_dump_json()])
    agent = ModelExplorer(
        router=ModelRouter(Settings(), available_providers={ProviderName.MOCK}),
        providers=ProviderRegistry([provider]),
        prompts=PromptRegistry(),
        max_retries=1,
    )
    input_data = ModelExplorerInput(analysis=analysis_fixture(), user_guidance=["Preserve Q1-Q3"])
    before = input_data.model_dump_json()
    result = await agent.run(
        input_data,
        ProblemState(
            project_id=uuid4(),
            title="Allocation",
            raw_problem="Allocation problem",
            subproblems=input_data.analysis.subproblems,
        ),
        TaskProfile(task_type=TaskType.MODEL_EXPLORATION),
    )

    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.is_mock is True
    assert result.attempts == 2 and result.token_usage.requests == 2
    assert diagnostic not in requests[0].messages[-1].content
    assert diagnostic in requests[1].messages[-1].content
    assert "AUTOMATED_RETRY_FEEDBACK" in requests[1].messages[-1].content
    assert "Preserve Q1-Q3" in requests[1].messages[-1].content
    assert input_data.model_dump_json() == before


def _data_blocked_exploration(available: int) -> ModelExploration:
    exploration = exploration_fixture()
    candidates = [
        candidate.model_copy(
            update={
                "data_requirements": [
                    requirement.model_copy(update={"availability": DataAvailability.MISSING})
                    for requirement in candidate.data_requirements
                ]
            }
        )
        if index >= available
        else candidate
        for index, candidate in enumerate(exploration.candidates)
    ]
    return exploration.model_copy(update={"candidates": candidates})


@pytest.mark.parametrize("available", [0, 1, 2])
def test_explore_gate_requires_two_data_executable_candidates(available: int) -> None:
    exploration = _data_blocked_exploration(available)
    before = exploration.model_dump_json()
    gate = explore_quality_gate(exploration, {"Q1", "Q2", "Q3"})

    assert (gate.status is QualityGateStatus.PASS) == (available >= 2)
    assert gate.checks["two_candidates_without_missing_required_data"] == (available >= 2)
    if available < 2:
        assert any(error.startswith("MISSING_REQUIRED_DATA:") for error in gate.errors)
    assert exploration.model_dump_json() == before


def test_explore_requires_the_same_two_candidates_to_have_data_and_full_coverage() -> None:
    exploration = _data_blocked_exploration(2)
    candidates = list(exploration.candidates)
    candidates[0] = candidates[0].model_copy(update={"target_subproblems": ["Q1"]})
    exploration = exploration.model_copy(update={"candidates": candidates})
    gate = explore_quality_gate(exploration, {"Q1", "Q2", "Q3"})
    assert gate.checks["two_candidates_without_missing_required_data"]
    assert not gate.checks["two_end_to_end_candidates"]
    assert gate.status is QualityGateStatus.RETRY


@pytest.mark.asyncio
@pytest.mark.parametrize("corrected", [True, False])
async def test_explorer_gate_feedback_is_bounded_and_cannot_accept_missing_data(
    corrected: bool,
) -> None:
    requests: list[GenerationRequest] = []

    class RecordingProvider(MockProvider):
        async def generate(self, request: GenerationRequest) -> ModelResponse:
            requests.append(request)
            return await super().generate(request)

    blocked = _data_blocked_exploration(0)
    next_output = exploration_fixture() if corrected else blocked
    provider = RecordingProvider([blocked.model_dump_json(), next_output.model_dump_json()])
    agent = ModelExplorer(
        router=ModelRouter(Settings(), available_providers={ProviderName.MOCK}),
        providers=ProviderRegistry([provider]),
        prompts=PromptRegistry(),
        max_retries=1,
    )
    analysis = analysis_fixture()
    state = ProblemState(
        project_id=uuid4(),
        title="Allocation",
        raw_problem="Allocation problem",
        subproblems=analysis.subproblems,
    )
    before = state.model_dump_json()
    result = await agent.run(
        ModelExplorerInput(analysis=analysis),
        state,
        TaskProfile(task_type=TaskType.MODEL_EXPLORATION),
    )

    assert result.attempts == 2
    assert result.is_mock is True
    assert "two_candidates_without_missing_required_data" in requests[1].messages[-1].content
    assert "never merely relabel" in requests[1].messages[-1].content
    assert (result.status is AgentRunStatus.SUCCEEDED) == corrected
    assert (result.output is not None) == corrected
    assert state.model_dump_json() == before
