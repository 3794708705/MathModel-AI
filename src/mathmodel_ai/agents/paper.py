from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.paper import (
    PaperAgentInput,
    PaperFactualAuditDraft,
    PaperFactualAuditInput,
    PaperIR,
)
from mathmodel_ai.schemas.problem_state import ProblemState


class PaperAgent(BaseAgent[PaperAgentInput, PaperIR]):
    name = "paper_agent"
    role = "evidence-grounded PaperIR authoring"
    capabilities = frozenset({"structured_generation", "paper_ir", "claim_authoring"})
    input_schema = PaperAgentInput
    output_schema = PaperIR

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
        input_data: PaperAgentInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[PaperIR]:
        del state
        prompt = self._prompts.get("paper_agent")
        response = await provider.structured_generate(
            GenerationRequest(
                model=route.selected_model or "unselected",
                reasoning_effort=route.selected_reasoning,
                messages=[
                    ModelMessage(role="system", content=prompt.system),
                    ModelMessage(
                        role="user",
                        content=prompt.render_user(
                            paper_input_json=input_data.model_dump_json(indent=2)
                        ),
                    ),
                ],
                metadata={"agent": self.name, "prompt_version": prompt.version},
            ),
            PaperIR,
        )
        return AgentExecution(
            output=response.parsed,
            response=response.response,
            prompt_version=prompt.version,
        )


class PaperFactualAuditAgent(BaseAgent[PaperFactualAuditInput, PaperFactualAuditDraft]):
    name = "paper_factual_audit_agent"
    role = "independent final paper factual audit"
    capabilities = frozenset({"structured_review", "factual_audit"})
    input_schema = PaperFactualAuditInput
    output_schema = PaperFactualAuditDraft

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
        input_data: PaperFactualAuditInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[PaperFactualAuditDraft]:
        del state
        prompt = self._prompts.get("paper_factual_audit_agent")
        response = await provider.structured_generate(
            GenerationRequest(
                model=route.selected_model or "unselected",
                reasoning_effort=route.selected_reasoning,
                messages=[
                    ModelMessage(role="system", content=prompt.system),
                    ModelMessage(
                        role="user",
                        content=prompt.render_user(
                            audit_input_json=input_data.model_dump_json(indent=2)
                        ),
                    ),
                ],
                metadata={"agent": self.name, "prompt_version": prompt.version},
            ),
            PaperFactualAuditDraft,
        )
        return AgentExecution(
            output=response.parsed,
            response=response.response,
            prompt_version=prompt.version,
        )
