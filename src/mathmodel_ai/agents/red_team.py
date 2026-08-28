import json

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.verification import RedTeamDraft, RedTeamInput


class RedTeamAgent(BaseAgent[RedTeamInput, RedTeamDraft]):
    name = "red_team_agent"
    role = "adversarial mathematical-model and result review"
    capabilities = frozenset({"structured_review", "adversarial_analysis", "claim_attack"})
    input_schema = RedTeamInput
    output_schema = RedTeamDraft

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
        input_data: RedTeamInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[RedTeamDraft]:
        prompt = self._prompts.get("red_team_agent")
        request = GenerationRequest(
            model=route.selected_model or "unselected",
            reasoning_effort=route.selected_reasoning,
            messages=[
                ModelMessage(role="system", content=prompt.system),
                ModelMessage(
                    role="user",
                    content=prompt.render_user(
                        mathematical_model_json=input_data.mathematical_model.model_dump_json(
                            indent=2
                        ),
                        validation_json=input_data.validation.model_dump_json(indent=2),
                        sensitivity_json=input_data.sensitivity.model_dump_json(indent=2),
                        robustness_json=input_data.robustness.model_dump_json(indent=2),
                        user_guidance=json.dumps(input_data.user_guidance, ensure_ascii=False),
                    ),
                ),
            ],
            metadata={"agent": self.name, "prompt_version": prompt.version},
        )
        response = await provider.structured_generate(request, RedTeamDraft)
        return AgentExecution(
            output=response.parsed,
            response=response.response,
            prompt_version=prompt.version,
        )
