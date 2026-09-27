import json
import re

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.mathematical.expressions import referenced_symbols
from mathmodel_ai.mathematical.normalization import (
    canonicalize_scalar_optimization_family,
    materialize_data_bound_scalars,
)
from mathmodel_ai.mathematical.quality_gates import model_quality_gate
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.mathematical import (
    ConstraintRelation,
    ExpressionKind,
    MathematicalModel,
    MathematicalModelDraft,
    MathematicalModelStatus,
    MathExpression,
    MathModelerInput,
    VariableDomain,
)
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.quality import QualityGateStatus

_VALIDATE_STAGE_CONTRACTS = frozenset(
    {
        "recompute variable bounds",
        "recompute every constraint",
        "recompute variable bounds and constraints",
        "recalculate objective metric",
        "verify evidence trace",
    }
)


def _single_residual_with_multiple_free_decisions(model: MathematicalModel) -> bool:
    """A single scalar residual cannot identify multiple free causal coefficients."""
    if model.objective is None:
        return False
    definitions: dict[str, MathExpression] = {}
    for variable in model.derived_variables:
        matches = [
            equation.rhs
            for equation in model.equations
            if equation.lhs.kind is ExpressionKind.SYMBOL and equation.lhs.symbol == variable.symbol
        ]
        if len(matches) == 1:
            definitions[variable.symbol] = matches[0]
    objective = next(
        (
            equation.rhs
            for equation in model.equations
            if equation.equation_id == model.objective.equation_ref
        ),
        model.objective.expression,
    )
    residual: MathExpression | None = None
    if (
        objective.kind is ExpressionKind.MULTIPLY
        and len(objective.operands) == 2
        and objective.operands[0] == objective.operands[1]
    ):
        residual = objective.operands[0]
    elif (
        objective.kind is ExpressionKind.POWER
        and len(objective.operands) == 2
        and objective.operands[1].kind is ExpressionKind.CONSTANT
        and objective.operands[1].value == 2
    ):
        residual = objective.operands[0]
    if residual is None or residual.kind is not ExpressionKind.SUBTRACT:
        return False
    free = {
        item.symbol
        for item in model.decision_variables
        if not item.index_sets
        and item.domain in {VariableDomain.CONTINUOUS, VariableDomain.NONNEGATIVE_CONTINUOUS}
        and (
            item.lower_bound is None
            or item.upper_bound is None
            or item.lower_bound < item.upper_bound
        )
    }
    if len(free) < 2:
        return False

    def dependencies(expression: MathExpression, visited: frozenset[str] = frozenset()) -> set[str]:
        result: set[str] = set()
        for symbol in referenced_symbols(expression):
            if symbol in free:
                result.add(symbol)
            elif symbol in definitions and symbol not in visited:
                result.update(dependencies(definitions[symbol], visited | {symbol}))
        return result

    # Independent equality constraints may identify additional coefficients;
    # inequality probability bounds alone cannot supply missing observations.
    if any(
        item.relation is ConstraintRelation.EQ
        and (dependencies(item.expression) or dependencies(item.rhs))
        for item in [*model.constraints, *model.initial_conditions, *model.boundary_conditions]
    ):
        return False
    return len(set().union(*(dependencies(item) for item in residual.operands))) >= 2


def reject_unverifiable_causal_requirements(model: MathematicalModel, guidance: list[str]) -> None:
    """Fail before solving when a causal model declares unverifiable checks."""
    protocol = next(
        (item for item in guidance if item.startswith("CAUSAL_HOLDOUT_PROTOCOL:")), None
    )
    if protocol is None:
        return
    allowed = set(re.findall(r"causal_science:[a-z_]+", protocol))
    allowed.update(_VALIDATE_STAGE_CONTRACTS)
    unknown = [
        requirement
        for requirement in model.validation_requirements
        if " ".join(requirement.casefold().split()) not in allowed
    ]
    if unknown:
        raise QualityGateError(
            "MODEL_GATE_FAIL:UNVERIFIABLE_VALIDATION_REQUIREMENT: "
            + " | ".join(item[:180] for item in unknown)
        )
    if _single_residual_with_multiple_free_decisions(model):
        raise QualityGateError("MODEL_GATE_FAIL:CAUSAL_SINGLE_RESIDUAL_UNDERIDENTIFIED")
    # Automatic causal runs have no pre-reviewed perturbation replay. An
    # objective-free, stateless model therefore enters the formal scalar
    # response experiment path, which must recompute every declared response.
    if model.objective is None and not model.state_variables:
        if not model.derived_variables or any(
            item.index_sets for item in [*model.decision_variables, *model.derived_variables]
        ):
            raise QualityGateError("MODEL_GATE_FAIL:CAUSAL_SCALAR_RESPONSE_UNSUPPORTED_STRUCTURE")
        definitions = {item.symbol: 0 for item in model.derived_variables}
        for equation in model.equations:
            if equation.lhs.kind is ExpressionKind.SYMBOL and equation.lhs.symbol in definitions:
                assert equation.lhs.symbol is not None
                definitions[equation.lhs.symbol] += 1
        incomplete = sorted(symbol for symbol, count in definitions.items() if count != 1)
        if incomplete:
            raise QualityGateError(
                "MODEL_GATE_FAIL:CAUSAL_SCALAR_RESPONSE_INCOMPLETE:" + ",".join(incomplete)
            )


class MathModeler(BaseAgent[MathModelerInput, MathematicalModel]):
    """Build an executable, source-typed model without inventing empirical calibration."""

    name = "math_modeler"
    role = "structured solver-independent mathematical model construction"
    capabilities = frozenset(
        {"structured_generation", "mathematical_modeling", "traceable_equations"}
    )
    input_schema = MathModelerInput
    output_schema = MathematicalModel

    def __init__(
        self,
        *,
        router: ModelRouter,
        providers: ProviderRegistry,
        prompts: PromptRegistry,
        max_retries: int = 2,
    ) -> None:
        super().__init__(router=router, providers=providers, max_retries=max_retries)
        self._prompts = prompts

    def prepare_attempt_input(
        self,
        input_data: MathModelerInput,
        state: ProblemState,
        previous_errors: tuple[str, ...],
    ) -> MathModelerInput:
        if not previous_errors:
            return input_data
        feedback = " | ".join(dict.fromkeys(error[:1200] for error in previous_errors[-3:]))
        return input_data.model_copy(
            update={
                "user_guidance": [
                    *input_data.user_guidance,
                    (
                        "AUTOMATED_RETRY_FEEDBACK: The previous draft was rejected. "
                        "Correct every listed issue without weakening or bypassing the "
                        f"deterministic gates: {feedback}"
                    ),
                ]
            }
        )

    def validate_output_for_state(
        self,
        output: MathematicalModel,
        state: ProblemState,
    ) -> MathematicalModel:
        gate = model_quality_gate(output, state)
        if gate.status is not QualityGateStatus.PASS:
            raise QualityGateError(f"MODEL quality gate rejected output: {', '.join(gate.errors)}")
        return output

    async def execute(
        self,
        input_data: MathModelerInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[MathematicalModel]:
        prompt = self._prompts.get("math_modeler")
        request = GenerationRequest(
            model=route.selected_model or "unselected",
            reasoning_effort=route.selected_reasoning,
            # The schema is intentionally rich and reasoning tokens count toward
            # OpenAI-compatible max_tokens on providers such as DeepSeek.
            max_output_tokens=65_536,
            messages=[
                ModelMessage(role="system", content=prompt.system),
                ModelMessage(
                    role="user",
                    content=prompt.render_user(
                        selected_model_json=input_data.selected_model.model_dump_json(indent=2),
                        problem_analysis_json=input_data.problem_analysis.model_dump_json(indent=2),
                        data_understanding_json=(
                            input_data.data_understanding.model_dump_json(indent=2)
                            if input_data.data_understanding is not None
                            else "null"
                        ),
                        data_profiles_json=json.dumps(
                            [item.model_dump(mode="json") for item in input_data.data_profiles],
                            ensure_ascii=False,
                            indent=2,
                        ),
                        user_guidance=json.dumps(input_data.user_guidance, ensure_ascii=False),
                    ),
                ),
            ],
            metadata={"agent": self.name, "prompt_version": prompt.version},
        )
        response = await provider.structured_generate(request, MathematicalModelDraft)
        model = MathematicalModel(
            **response.parsed.model_dump(),
            model_id=input_data.assigned_model_id,
            project_id=state.project_id,
            problem_id=state.problem_id,
            version=input_data.assigned_version,
            source_selected_model_id=input_data.selected_model.candidate_id,
            status=MathematicalModelStatus.READY,
        )
        model = canonicalize_scalar_optimization_family(model)
        model = materialize_data_bound_scalars(model, state)
        reject_unverifiable_causal_requirements(model, input_data.user_guidance)
        return AgentExecution(
            output=model,
            response=response.response,
            prompt_version=prompt.version,
        )
