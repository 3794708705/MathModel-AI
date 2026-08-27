import json
import re

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.execution import ExecutionOrigin
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.program import (
    CodeAgentInput,
    GeneratedProgram,
    GeneratedProgramDraft,
    GeneratedProgramStatus,
    generated_program_hash,
)

_HARDCODED_RESULT_PATTERNS = (
    re.compile(
        r"print\s*\(\s*['\"]\s*(?:objective|result)\s*=\s*[-+]?\d+(?:\.\d+)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"['\"](?:objective|objective_value|result)['\"]\s*:\s*[-+]?\d+(?:\.\d+)?",
        re.IGNORECASE,
    ),
)


def reject_hardcoded_results(program: GeneratedProgramDraft) -> None:
    for source in program.files:
        if any(pattern.search(source.content) for pattern in _HARDCODED_RESULT_PATTERNS):
            raise ValueError(
                f"CODE_GENERATION_BLOCKED: apparent hard-coded result in {source.path}"
            )


class CodeAgent(BaseAgent[CodeAgentInput, GeneratedProgram]):
    name = "code_agent"
    role = "auditable translation of a fixed mathematical model into executable code"
    capabilities = frozenset({"structured_generation", "code_generation"})
    input_schema = CodeAgentInput
    output_schema = GeneratedProgram

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
        input_data: CodeAgentInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[GeneratedProgram]:
        prompt = self._prompts.get("code_agent")
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
                        model_digest=mathematical_model_digest(input_data.mathematical_model),
                        algorithm_plan_json=input_data.algorithm_plan.model_dump_json(indent=2),
                        user_guidance=json.dumps(input_data.user_guidance, ensure_ascii=False),
                    ),
                ),
            ],
            metadata={"agent": self.name, "prompt_version": prompt.version},
        )
        response = await provider.structured_generate(request, GeneratedProgramDraft)
        reject_hardcoded_results(response.parsed)
        files = [item.with_digest() for item in response.parsed.files]
        program = GeneratedProgram(
            **response.parsed.model_dump(exclude={"files"}),
            files=files,
            project_id=state.project_id,
            problem_id=state.problem_id,
            model_id=input_data.mathematical_model.model_id,
            model_version=input_data.mathematical_model.version,
            model_digest=mathematical_model_digest(input_data.mathematical_model),
            code_hash=generated_program_hash(files),
            generated_by=self.name,
            prompt_version=prompt.version,
            execution_origin=ExecutionOrigin.GENERATED_PROGRAM,
            status=GeneratedProgramStatus.READY,
            is_mock=route.selected_provider is not None and route.selected_provider.value == "mock",
        )
        return AgentExecution(
            output=program,
            response=response.response,
            prompt_version=prompt.version,
        )
