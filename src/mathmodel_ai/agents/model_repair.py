import json

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.mathematical import MathematicalModel, MathematicalModelStatus
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.verification import (
    ModelRepairDraft,
    ModelRepairInput,
    ModelRepairOutput,
)


class ModelRepairAgent(BaseAgent[ModelRepairInput, ModelRepairOutput]):
    name = "model_repair_agent"
    role = "finding-scoped immutable mathematical-model repair"
    capabilities = frozenset({"structured_generation", "model_repair", "finding_resolution"})
    input_schema = ModelRepairInput
    output_schema = ModelRepairOutput

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
        input_data: ModelRepairInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[ModelRepairOutput]:
        prompt = self._prompts.get("model_repair_agent")
        request = GenerationRequest(
            model=route.selected_model or "unselected",
            reasoning_effort=route.selected_reasoning,
            messages=[
                ModelMessage(role="system", content=prompt.system),
                ModelMessage(
                    role="user",
                    content=prompt.render_user(
                        current_model_json=input_data.current_model.model_dump_json(indent=2),
                        red_team_json=input_data.red_team_report.model_dump_json(indent=2),
                        validation_json=input_data.validation.model_dump_json(indent=2),
                        sensitivity_json=input_data.sensitivity.model_dump_json(indent=2),
                        robustness_json=input_data.robustness.model_dump_json(indent=2),
                        assigned_version=str(input_data.assigned_version),
                        repair_cycle=str(input_data.repair_cycle),
                        user_guidance=json.dumps(input_data.user_guidance, ensure_ascii=False),
                    ),
                ),
            ],
            metadata={"agent": self.name, "prompt_version": prompt.version},
        )
        response = await provider.structured_generate(request, ModelRepairDraft)
        draft = response.parsed
        repaired = MathematicalModel(
            **draft.revised_model.model_dump(),
            model_id=input_data.current_model.model_id,
            project_id=input_data.current_model.project_id,
            problem_id=input_data.current_model.problem_id,
            version=input_data.assigned_version,
            source_selected_model_id=input_data.current_model.source_selected_model_id,
            status=MathematicalModelStatus.READY,
        )
        return AgentExecution(
            output=ModelRepairOutput(
                revised_model=repaired,
                actions=draft.actions,
                addressed_finding_ids=draft.addressed_finding_ids,
                remaining_risks=draft.remaining_risks,
            ),
            response=response.response,
            prompt_version=prompt.version,
        )
