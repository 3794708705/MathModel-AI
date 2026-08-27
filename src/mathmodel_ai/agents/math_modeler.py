import json

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
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


class MathModeler(BaseAgent[MathModelerInput, MathematicalModel]):
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
        return AgentExecution(
            output=model,
            response=response.response,
            prompt_version=prompt.version,
        )
