import json

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.problem_analysis import (
    AmbiguityReviewStatus,
    ProblemAgentInput,
    ProblemAnalysis,
)
from mathmodel_ai.schemas.problem_state import ProblemState


class ProblemAgent(BaseAgent[ProblemAgentInput, ProblemAnalysis]):
    name = "problem_agent"
    role = "structured problem understanding and decomposition"
    capabilities = frozenset({"structured_generation", "problem_decomposition"})
    input_schema = ProblemAgentInput
    output_schema = ProblemAnalysis

    def __init__(
        self,
        *,
        router: ModelRouter,
        providers: ProviderRegistry,
        prompts: PromptRegistry,
        max_retries: int = 2,
        ambiguity_review_threshold: float = 0.7,
    ) -> None:
        super().__init__(router=router, providers=providers, max_retries=max_retries)
        self._prompts = prompts
        self._ambiguity_review_threshold = ambiguity_review_threshold

    def prepare_attempt_input(
        self,
        input_data: ProblemAgentInput,
        state: ProblemState,
        previous_errors: tuple[str, ...],
    ) -> ProblemAgentInput:
        del state
        if not previous_errors:
            return input_data
        return input_data.model_copy(
            update={
                "repair_feedback": [
                    *input_data.repair_feedback[:2], previous_errors[-1][:4096]
                ]
            }
        )

    async def execute(
        self,
        input_data: ProblemAgentInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[ProblemAnalysis]:
        del state
        prompt = self._prompts.get("problem_agent")
        request = GenerationRequest(
            model=route.selected_model or "unselected",
            reasoning_effort=route.selected_reasoning,
            messages=[
                ModelMessage(role="system", content=prompt.system),
                ModelMessage(
                    role="user",
                    content=prompt.render_user(
                        title=input_data.title,
                        competition_context=input_data.competition_context or "not provided",
                        user_notes=json.dumps(input_data.optional_user_notes, ensure_ascii=False),
                        repair_feedback=json.dumps(input_data.repair_feedback, ensure_ascii=False),
                        raw_problem=input_data.raw_problem,
                    ),
                ),
            ],
            metadata={"agent": self.name, "prompt_version": prompt.version},
        )
        response = await provider.structured_generate(request, ProblemAnalysis)
        analysis = response.parsed
        ambiguities = []
        review_recommended = analysis.human_review_recommended or any(
            item.review_status is AmbiguityReviewStatus.HUMAN_REVIEW_RECOMMENDED
            for item in analysis.ambiguities
        )
        for ambiguity in analysis.ambiguities:
            if ambiguity.confidence < self._ambiguity_review_threshold:
                ambiguity = ambiguity.model_copy(
                    update={"review_status": AmbiguityReviewStatus.HUMAN_REVIEW_RECOMMENDED}
                )
                review_recommended = True
            ambiguities.append(ambiguity)
        analysis = analysis.model_copy(
            update={
                "ambiguities": ambiguities,
                "human_review_recommended": review_recommended,
            }
        )
        return AgentExecution(
            output=analysis,
            response=response.response,
            prompt_version=prompt.version,
        )
