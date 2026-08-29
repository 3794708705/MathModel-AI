import json

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.paper import LiteratureAgentInput, LiteraturePlan
from mathmodel_ai.schemas.problem_state import ProblemState


class LiteratureAgent(BaseAgent[LiteratureAgentInput, LiteraturePlan]):
    name = "literature_agent"
    role = "evidence-oriented literature search planning"
    capabilities = frozenset({"structured_generation", "literature_planning"})
    input_schema = LiteratureAgentInput
    output_schema = LiteraturePlan

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
        input_data: LiteratureAgentInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[LiteraturePlan]:
        del state
        prompt = self._prompts.get("literature_agent")
        response = await provider.structured_generate(
            GenerationRequest(
                model=route.selected_model or "unselected",
                reasoning_effort=route.selected_reasoning,
                messages=[
                    ModelMessage(role="system", content=prompt.system),
                    ModelMessage(
                        role="user",
                        content=prompt.render_user(
                            problem_title=input_data.problem_title,
                            problem_summary=input_data.problem_summary,
                            model_name=input_data.model_name,
                            model_family=input_data.model_family,
                            evidence_summaries=json.dumps(
                                input_data.evidence_summaries, ensure_ascii=False
                            ),
                            requested_uses=json.dumps(
                                [item.value for item in input_data.requested_uses]
                            ),
                        ),
                    ),
                ],
                metadata={"agent": self.name, "prompt_version": prompt.version},
            ),
            LiteraturePlan,
        )
        return AgentExecution(
            output=response.parsed,
            response=response.response,
            prompt_version=prompt.version,
        )
