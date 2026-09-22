from pathlib import Path

import pytest
from pydantic import ValidationError

from mathmodel_ai.agents import AgentRunStatus, CodeAgent, MathModeler
from mathmodel_ai.core.config import Settings
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.providers.schemas import GenerationRequest, ModelResponse
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import EscalationLevel, TaskProfile, TaskType
from mathmodel_ai.schemas.mathematical import (
    ConvexityStatus,
    MathematicalModel,
    MathematicalModelDraft,
    MathModelerInput,
    VariableDomain,
)
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.program import CodeAgentInput, GeneratedProgramDraft, GeneratedSourceFile
from mathmodel_ai.schemas.solver import AlgorithmFamily, SolverFamily
from tests.mathematical.helpers import lp_model, milp_model, nlp_model, selected_state


class RecordingMockProvider(MockProvider):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.last_request: GenerationRequest | None = None
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        self.last_request = request
        self.requests.append(request)
        return await super().generate(request)


def _agent_services(mock: MockProvider) -> dict[str, object]:
    settings = Settings(default_provider="mock", reasoning_max_retries=0)
    providers = ProviderRegistry([mock])
    return {
        "router": ModelRouter(settings, available_providers=providers.available),
        "providers": providers,
        "prompts": PromptRegistry(),
        "max_retries": 0,
    }


def test_algorithm_selector_uses_declared_family_and_configurable_preferences() -> None:
    selector = AlgorithmSelector()
    lp = selector.select(lp_model())
    milp = selector.select(milp_model())
    nlp = selector.select(nlp_model())

    assert lp.algorithm_family is AlgorithmFamily.LINEAR_OPTIMIZATION
    assert lp.recommended_solver_family is SolverFamily.SCIPY_HIGHS
    assert milp.algorithm_family is AlgorithmFamily.MIXED_INTEGER_OPTIMIZATION
    assert milp.recommended_solver_family is SolverFamily.GUROBI
    assert SolverFamily.SCIPY_MILP in milp.alternatives
    assert nlp.algorithm_family is AlgorithmFamily.NONLINEAR_LOCAL_OPTIMIZATION
    assert nlp.recommended_solver_family is SolverFamily.SCIPY_MINIMIZE

    custom = AlgorithmSelector(
        {milp_model().model_family: [SolverFamily.SCIPY_MILP, SolverFamily.GUROBI]}
    ).select(milp_model())
    assert custom.recommended_solver_family is SolverFamily.SCIPY_MILP


def test_algorithm_selector_handles_integer_binary_and_declared_solver_override() -> None:
    integer = milp_model().model_copy(update={"model_family": ModelFamily.INTEGER_PROGRAMMING})
    binary_variables = [
        item.model_copy(update={"domain": VariableDomain.BINARY, "upper_bound": 1})
        for item in integer.decision_variables
    ]
    binary = integer.model_copy(update={"decision_variables": binary_variables})
    requirements = binary.solver_requirements.model_copy(
        update={"preferred_solver_families": ["SCIPY_MILP"]}
    )
    overridden = binary.model_copy(update={"solver_requirements": requirements})

    integer_plan = AlgorithmSelector().select(integer)
    binary_plan = AlgorithmSelector().select(binary)
    overridden_plan = AlgorithmSelector().select(overridden)

    assert integer_plan.algorithm_family is AlgorithmFamily.INTEGER_CONSTRAINT_PROGRAMMING
    assert integer_plan.recommended_solver_family is SolverFamily.ORTOOLS_CP_SAT
    assert binary_plan.algorithm_family is AlgorithmFamily.INTEGER_CONSTRAINT_PROGRAMMING
    assert overridden_plan.recommended_solver_family is SolverFamily.SCIPY_MILP


def test_algorithm_selector_blocks_missing_objective_and_unsupported_mixed_integer_nlp() -> None:
    missing_objective = lp_model().model_copy(update={"objective": None})
    nlp = nlp_model()
    mixed_variables = [
        nlp.decision_variables[0].model_copy(update={"domain": VariableDomain.INTEGER}),
        *nlp.decision_variables[1:],
    ]
    mixed_integer_nlp = nlp.model_copy(update={"decision_variables": mixed_variables})

    missing_plan = AlgorithmSelector().select(missing_objective)
    mixed_plan = AlgorithmSelector().select(mixed_integer_nlp)

    assert missing_plan.algorithm_family is AlgorithmFamily.UNSUPPORTED
    assert missing_plan.recommended_solver_family is None
    assert "without an objective" in missing_plan.reason
    assert mixed_plan.algorithm_family is AlgorithmFamily.UNSUPPORTED
    assert mixed_plan.recommended_solver_family is None
    assert "mixed-integer nonlinear" in mixed_plan.reason


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        (ModelFamily.LEAST_SQUARES, AlgorithmFamily.LEAST_SQUARES),
        (ModelFamily.GRAPH, AlgorithmFamily.GRAPH_ALGORITHM),
        (ModelFamily.NETWORK_FLOW, AlgorithmFamily.GRAPH_ALGORITHM),
        (ModelFamily.SIMULATION, AlgorithmFamily.SIMULATION),
        (ModelFamily.DISCRETE_EVENT, AlgorithmFamily.SIMULATION),
        (ModelFamily.STATISTICAL, AlgorithmFamily.STATISTICAL_ESTIMATION),
        (ModelFamily.REGRESSION, AlgorithmFamily.STATISTICAL_ESTIMATION),
        (ModelFamily.OTHER, AlgorithmFamily.UNSUPPORTED),
    ],
)
def test_algorithm_selector_classifies_non_adapter_families_without_keyword_guessing(
    family: ModelFamily,
    expected: AlgorithmFamily,
) -> None:
    plan = AlgorithmSelector().select(lp_model().model_copy(update={"model_family": family}))
    assert plan.algorithm_family is expected
    assert plan.recommended_solver_family is None


def test_selector_rejects_invalid_preferences_and_domains_and_warns_for_nlp() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        AlgorithmSelector({ModelFamily.LINEAR_PROGRAMMING: []})
    lp = lp_model()
    integer_lp = lp.model_copy(
        update={
            "decision_variables": [
                lp.decision_variables[0].model_copy(update={"domain": VariableDomain.INTEGER}),
                *lp.decision_variables[1:],
            ]
        }
    )
    invalid_plan = AlgorithmSelector().select(integer_lp)
    nlp = nlp_model()
    unknown_requirements = nlp.algorithm_requirements.model_copy(
        update={"convexity": ConvexityStatus.UNKNOWN}
    )
    risky_plan = AlgorithmSelector().select(
        nlp.model_copy(update={"algorithm_requirements": unknown_requirements})
    )

    assert invalid_plan.algorithm_family is AlgorithmFamily.UNSUPPORTED
    assert "integer or binary" in invalid_plan.reason
    assert any("local solution" in risk for risk in risky_plan.numerical_risks)


@pytest.mark.asyncio
async def test_math_modeler_binds_identity_and_audits_xhigh_mock_route() -> None:
    state = selected_state()
    template = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    draft = MathematicalModelDraft.model_validate(
        template.model_dump(
            exclude={
                "model_id",
                "project_id",
                "problem_id",
                "version",
                "source_selected_model_id",
                "status",
            }
        )
    )
    mock = RecordingMockProvider([draft.model_dump_json()])
    agent = MathModeler(**_agent_services(mock))  # type: ignore[arg-type]
    assigned_model_id = template.model_id

    run = await agent.run(
        MathModelerInput(
            assigned_model_id=assigned_model_id,
            assigned_version=1,
            selected_model=state.selected_model,
            problem_analysis=state.problem_analysis,
        ),
        state,
        TaskProfile(
            task_type=TaskType.MATHEMATICAL_MODELING,
            minimum_level=EscalationLevel.FLAGSHIP_XHIGH,
        ),
    )

    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.output is not None
    assert run.output.model_id == assigned_model_id
    assert run.output.project_id == state.project_id
    assert run.output.source_selected_model_id == "CAND-lp"
    assert run.prompt_version == "4.3.0"
    assert mock.last_request is not None
    assert mock.last_request.max_output_tokens == 65_536
    assert run.routes[0].level is EscalationLevel.FLAGSHIP_XHIGH
    assert run.is_mock is True


@pytest.mark.asyncio
async def test_math_modeler_retries_state_gate_with_deterministic_feedback() -> None:
    state = selected_state()
    template = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    invalid = template.model_copy(update={"decision_variables": []})

    def draft_json(model: MathematicalModel) -> str:
        payload = model.model_dump(
            exclude={
                "model_id",
                "project_id",
                "problem_id",
                "version",
                "source_selected_model_id",
                "status",
            }
        )
        return MathematicalModelDraft.model_validate(payload).model_dump_json()

    mock = RecordingMockProvider([draft_json(invalid), draft_json(template)])
    services = _agent_services(mock)
    services["max_retries"] = 1
    agent = MathModeler(**services)  # type: ignore[arg-type]

    run = await agent.run(
        MathModelerInput(
            assigned_model_id=template.model_id,
            assigned_version=1,
            selected_model=state.selected_model,
            problem_analysis=state.problem_analysis,
        ),
        state,
        TaskProfile(
            task_type=TaskType.MATHEMATICAL_MODELING,
            minimum_level=EscalationLevel.FLAGSHIP_XHIGH,
        ),
    )

    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.attempts == 2
    assert any("MODEL_GATE_FAIL:decision_variables_present" in item for item in run.errors)
    assert len(mock.requests) == 2
    retry_prompt = mock.requests[1].messages[-1].content
    assert "AUTOMATED_RETRY_FEEDBACK" in retry_prompt
    assert "MODEL_GATE_FAIL:decision_variables_present" in retry_prompt


@pytest.mark.asyncio
async def test_code_agent_generates_hashed_program_without_changing_model() -> None:
    state = selected_state()
    model = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    plan = AlgorithmSelector().select(model)
    draft = GeneratedProgramDraft(
        entrypoint="solve.py",
        files=[
            GeneratedSourceFile(
                path="solve.py",
                content=(
                    "from pathlib import Path\n"
                    "value = sum([1, 2])\n"
                    "Path('/output/result.json').write_text(str(value))\n"
                ),
            )
        ],
        dependencies=[],
        solver_target="custom",
        explanation="Runtime computation fixture.",
    )
    mock = RecordingMockProvider([draft.model_dump_json()])
    agent = CodeAgent(**_agent_services(mock))  # type: ignore[arg-type]

    run = await agent.run(
        CodeAgentInput(mathematical_model=model, algorithm_plan=plan),
        state,
        TaskProfile(task_type=TaskType.CODE_GENERATION),
    )

    assert run.status is AgentRunStatus.SUCCEEDED
    assert run.output is not None
    assert run.output.model_id == model.model_id
    assert run.output.files[0].sha256 is not None
    assert len(run.output.code_hash) == 64
    assert run.output.is_mock is True
    assert '"title":"GeneratedResultPayload"' in mock.requests[0].messages[-1].content


@pytest.mark.asyncio
async def test_code_agent_blocks_obvious_hardcoded_result() -> None:
    state = selected_state()
    model = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    bad = GeneratedProgramDraft(
        entrypoint="solve.py",
        files=[GeneratedSourceFile(path="solve.py", content='print("objective = 123.45")')],
        dependencies=[],
        solver_target="custom",
        explanation="Invalid fixture.",
    )
    mock = MockProvider([bad.model_dump_json()])
    agent = CodeAgent(**_agent_services(mock))  # type: ignore[arg-type]

    run = await agent.run(
        CodeAgentInput(
            mathematical_model=model,
            algorithm_plan=AlgorithmSelector().select(model),
        ),
        state,
        TaskProfile(task_type=TaskType.CODE_GENERATION),
    )

    assert run.status is AgentRunStatus.ESCALATED
    assert any("CODE_GENERATION_BLOCKED" in error for error in run.errors)


def test_generated_program_rejects_solver_target_larger_than_database_contract() -> None:
    with pytest.raises(ValidationError, match="String should have at most 64 characters"):
        GeneratedProgramDraft(
            entrypoint="solve.py",
            files=[GeneratedSourceFile(path="solve.py", content="print('ok')")],
            solver_target="x" * 65,
            explanation="Invalid persistence contract fixture.",
        )


def test_versioned_prompt_resources_exist() -> None:
    prompts = PromptRegistry()
    assert prompts.get("math_modeler").version == "4.3.0"
    assert prompts.get("code_agent").version == "4.4.0"
    assert (
        "must report every MathematicalModel decision variable" in prompts.get("code_agent").system
    )
    assert Path("src/mathmodel_ai/prompt_templates/math_modeler.prompt").is_file()
