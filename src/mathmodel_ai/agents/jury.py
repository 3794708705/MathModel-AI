import json

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.reasoning.scoring import build_model_selection
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.model_selection import (
    JuryAssessment,
    ModelJuryInput,
    ModelJuryWeights,
    ModelSelection,
)
from mathmodel_ai.schemas.problem_state import ProblemState


class ModelJury(BaseAgent[ModelJuryInput, ModelSelection]):
    name = "model_jury"
    role = "qualitative review with deterministic model selection"
    capabilities = frozenset({"structured_generation", "model_review"})
    input_schema = ModelJuryInput
    output_schema = ModelSelection

    def __init__(
        self,
        *,
        router: ModelRouter,
        providers: ProviderRegistry,
        prompts: PromptRegistry,
        weights: ModelJuryWeights,
        max_retries: int = 2,
    ) -> None:
        super().__init__(router=router, providers=providers, max_retries=max_retries)
        self._prompts = prompts
        self._weights = weights

    async def execute(
        self,
        input_data: ModelJuryInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[ModelSelection]:
        del state
        prompt = self._prompts.get("model_jury")
        request = GenerationRequest(
            model=route.selected_model or "unselected",
            reasoning_effort=route.selected_reasoning,
            messages=[
                ModelMessage(role="system", content=prompt.system),
                ModelMessage(
                    role="user",
                    content=prompt.render_user(
                        analysis_json=input_data.analysis.model_dump_json(indent=2),
                        candidates_json=json.dumps(
                            [item.model_dump(mode="json") for item in input_data.candidates],
                            ensure_ascii=False,
                            indent=2,
                        ),
                        jury_notes=json.dumps(input_data.jury_notes, ensure_ascii=False),
                    ),
                ),
            ],
            metadata={"agent": self.name, "prompt_version": prompt.version},
        )
        response = await provider.structured_generate(request, JuryAssessment)
        selection = build_model_selection(
            input_data.candidates,
            response.parsed,
            self._weights,
            required_subproblem_ids={
                item.subproblem_id for item in input_data.analysis.subproblems
            },
        )
        return AgentExecution(
            output=selection,
            response=response.response,
            prompt_version=prompt.version,
        )
