import json

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.deduplication import deduplicate_candidates
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.reasoning.quality_gates import explore_quality_gate
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.model_selection import ModelExploration, ModelExplorerInput
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.quality import QualityGateStatus


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

    def prepare_attempt_input(
        self,
        input_data: ModelExplorerInput,
        state: ProblemState,
        previous_errors: tuple[str, ...],
    ) -> ModelExplorerInput:
        if not previous_errors:
            return input_data
        return input_data.model_copy(
            update={
                "user_guidance": [
                    *input_data.user_guidance,
                    (
                        "AUTOMATED_RETRY_FEEDBACK: The previous response was rejected. "
                        "Return a complete, concise JSON object matching the requested "
                        "schema. Correct JSON syntax and escaping at the reported location "
                        "and every schema error; preserve distinct candidates, required "
                        "subproblem coverage, and honest limitations. For missing-data "
                        "gate failures, reformulate the proposed implementation or supply "
                        "a justified acquisition/assumption plan; never merely relabel "
                        f"unavailable observations: {previous_errors[-1][:4096]}"
                    ),
                ]
            }
        )

    def validate_output_for_state(
        self, output: ModelExploration, state: ProblemState
    ) -> ModelExploration:
        gate = explore_quality_gate(output, {item.subproblem_id for item in state.subproblems})
        if gate.status is not QualityGateStatus.PASS:
            raise QualityGateError(
                f"EXPLORE quality gate rejected output: {', '.join(gate.errors)}"
            )
        return output

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
