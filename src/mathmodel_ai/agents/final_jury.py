from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.submission import FinalJuryDraft, FinalJuryInput


class FinalJuryAgent(BaseAgent[FinalJuryInput, FinalJuryDraft]):
    name = "final_jury_agent"
    role = "independent competition submission reviewer"
    capabilities = frozenset({"structured_review", "final_jury", "competition_review"})
    input_schema = FinalJuryInput
    output_schema = FinalJuryDraft

    def __init__(
        self,
        *,
        router: ModelRouter,
        providers: ProviderRegistry,
        prompts: PromptRegistry,
        max_retries: int = 1,
    ) -> None:
        super().__init__(router=router, providers=providers, max_retries=max_retries)
        self._prompts = prompts

    async def execute(
        self,
        input_data: FinalJuryInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[FinalJuryDraft]:
        del state
        prompt = self._prompts.get("final_jury_agent")
        response = await provider.structured_generate(
            GenerationRequest(
                model=route.selected_model or "unselected",
                reasoning_effort=route.selected_reasoning,
                messages=[
                    ModelMessage(role="system", content=prompt.system),
                    ModelMessage(
                        role="user",
                        content=prompt.render_user(
                            final_jury_input_json=input_data.model_dump_json(indent=2)
                        ),
                    ),
                ],
                metadata={"agent": self.name, "prompt_version": prompt.version},
            ),
            FinalJuryDraft,
        )
        return AgentExecution(
            output=response.parsed,
            response=response.response,
            prompt_version=prompt.version,
        )
