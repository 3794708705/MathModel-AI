from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.paper import CitationAgentInput, CitationSupportDraft
from mathmodel_ai.schemas.problem_state import ProblemState


class CitationAgent(BaseAgent[CitationAgentInput, CitationSupportDraft]):
    name = "citation_agent"
    role = "claim-to-source support review"
    capabilities = frozenset({"structured_review", "citation_support"})
    input_schema = CitationAgentInput
    output_schema = CitationSupportDraft

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
        input_data: CitationAgentInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[CitationSupportDraft]:
        del state
        prompt = self._prompts.get("citation_agent")
        response = await provider.structured_generate(
            GenerationRequest(
                model=route.selected_model or "unselected",
                reasoning_effort=route.selected_reasoning,
                messages=[
                    ModelMessage(role="system", content=prompt.system),
                    ModelMessage(
                        role="user",
                        content=prompt.render_user(
                            claim_json=input_data.claim.model_dump_json(indent=2),
                            reference_json=input_data.reference.model_dump_json(indent=2),
                            trusted_text=input_data.trusted_text,
                        ),
                    ),
                ],
                metadata={"agent": self.name, "prompt_version": prompt.version},
            ),
            CitationSupportDraft,
        )
        return AgentExecution(
            output=response.parsed,
            response=response.response,
            prompt_version=prompt.version,
        )
