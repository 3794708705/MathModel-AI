import json
import re

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.mathematical.normalization import canonicalize_scalar_optimization_family
from mathmodel_ai.mathematical.quality_gates import model_quality_gate
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.mathematical import (
    MathematicalModel,
    MathematicalModelDraft,
    MathematicalModelStatus,
    MathModelerInput,
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
        feedback = previous_errors[-1][:4096]
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
        reject_unverifiable_causal_requirements(model, input_data.user_guidance)
        return AgentExecution(
            output=model,
            response=response.response,
            prompt_version=prompt.version,
        )
