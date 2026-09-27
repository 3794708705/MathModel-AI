import json

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.core.errors import ProviderResponseError
from mathmodel_ai.data.quality_gates import data_column_reference_errors
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, MediaPart, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.data import DataAgentInput, DataUnderstanding
from mathmodel_ai.schemas.problem_state import ProblemState


class DataAgent(BaseAgent[DataAgentInput, DataUnderstanding]):
    name = "data_agent"
    role = "structured interpretation of deterministic data profiles and media"
    capabilities = frozenset(
        {"structured_generation", "data_understanding", "multimodal_understanding"}
    )
    input_schema = DataAgentInput
    output_schema = DataUnderstanding

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
        input_data: DataAgentInput,
        state: ProblemState,
        previous_errors: tuple[str, ...],
    ) -> DataAgentInput:
        del state
        if not previous_errors:
            return input_data
        return input_data.model_copy(
            update={"user_guidance": [*input_data.user_guidance, previous_errors[-1]]}
        )

    async def execute(
        self,
        input_data: DataAgentInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[DataUnderstanding]:
        del state
        prompt = self._prompts.get("data_agent")
        media = [
            MediaPart(
                mime_type=item.mime_type,
                data_base64=item.data_base64,
                source_id=str(item.file_id),
            )
            for item in input_data.media_assets
        ]
        request = GenerationRequest(
            model=route.selected_model or "unselected",
            reasoning_effort=route.selected_reasoning,
            messages=[
                ModelMessage(role="system", content=prompt.system),
                ModelMessage(
                    role="user",
                    content=prompt.render_user(
                        raw_problem=input_data.raw_problem,
                        problem_analysis_json=(
                            input_data.problem_analysis.model_dump_json(indent=2)
                            if input_data.problem_analysis is not None
                            else "null"
                        ),
                        profiles_json=json.dumps(
                            [item.model_dump(mode="json") for item in input_data.profiles],
                            ensure_ascii=False,
                            indent=2,
                        ),
                        relationships_json=json.dumps(
                            [
                                item.model_dump(mode="json")
                                for item in input_data.cross_dataset_relationships
                            ],
                            ensure_ascii=False,
                            indent=2,
                        ),
                        user_guidance=json.dumps(input_data.user_guidance, ensure_ascii=False),
                    ),
                    media=media,
                ),
            ],
            metadata={"agent": self.name, "prompt_version": prompt.version},
        )
        response = await provider.structured_generate(request, DataUnderstanding)
        output = response.parsed
        expected_ids = {profile.dataset_id for profile in input_data.profiles}
        actual_ids = {dataset.dataset_id for dataset in output.datasets}
        errors = data_column_reference_errors(input_data.profiles, output)
        if actual_ids != expected_ids:
            errors.insert(
                0,
                "dataset IDs must exactly match supplied profiles; "
                f"missing={sorted(map(str, expected_ids - actual_ids))}; "
                f"unknown={sorted(map(str, actual_ids - expected_ids))}",
            )
        if errors:
            raise ProviderResponseError("DATA profile binding failed: " + "; ".join(errors[:8]))
        return AgentExecution(
            output=output,
            response=response.response,
            prompt_version=prompt.version,
        )
