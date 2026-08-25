import json

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.deduplication import deduplicate_candidates
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.model_selection import ModelExploration, ModelExplorerInput
from mathmodel_ai.schemas.problem_state import ProblemState


class ModelExplorer(BaseAgent[ModelExplorerInput, ModelExploration]):
    name = "model_explorer"
    role = "generation and comparison of distinct model candidates"
    capabilities = frozenset({"structured_generation", "model_exploration"})
    input_schema = ModelExplorerInput
    output_schema = ModelExploration

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
        input_data: ModelExplorerInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[ModelExploration]:
        del state
        prompt = self._prompts.get("model_explorer")
        request = GenerationRequest(
            model=route.selected_model or "unselected",
            reasoning_effort=route.selected_reasoning,
            messages=[
                ModelMessage(role="system", content=prompt.system),
                ModelMessage(
                    role="user",
                    content=prompt.render_user(
                        analysis_json=input_data.analysis.model_dump_json(indent=2),
                        user_guidance=json.dumps(input_data.user_guidance, ensure_ascii=False),
                    ),
                ),
            ],
            metadata={"agent": self.name, "prompt_version": prompt.version},
        )
        response = await provider.structured_generate(request, ModelExploration)
        deduplicated = deduplicate_candidates(response.parsed.candidates)
        reason = response.parsed.fewer_than_three_reason
        if len(deduplicated.candidates) < 3 and not reason:
            reason = (
                "Semantic duplicate removal left fewer than three genuinely distinct "
                f"candidates: {', '.join(deduplicated.removed_ids)}"
            )
        exploration = ModelExploration(
            candidates=deduplicated.candidates,
            fewer_than_three_reason=reason,
            exploration_summary=response.parsed.exploration_summary,
        )
        return AgentExecution(
            output=exploration,
            response=response.response,
            prompt_version=prompt.version,
        )
