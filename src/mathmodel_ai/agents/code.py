import ast
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
from mathmodel_ai.schemas.solver import GeneratedResultPayload

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


def reject_header_only_csv_usage(program: GeneratedProgramDraft) -> None:
    """Reject the observed failure mode of treating a CSV header as the data."""
    readers: set[str] = set()
    trees: list[ast.AST] = []
    for source in program.files:
        try:
            module = ast.parse(source.content, filename=source.path)
        except SyntaxError as exc:
            raise ValueError(f"CODE_GENERATION_BLOCKED: invalid Python in {source.path}") from exc
        trees.append(module)
        for node in ast.walk(module):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr in {"reader", "DictReader"}
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            readers.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
    if not readers:
        return

    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and any(
                isinstance(item, ast.Name) and item.id in readers
                for item in ast.walk(node.iter)
            ):
                return
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "read_csv",
                "scan_csv",
            } and not any(
                keyword.arg == "nrows"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 0
                for keyword in node.keywords
            ):
                return
            if (
                any(isinstance(arg, ast.Name) and arg.id in readers for arg in node.args)
                and not (isinstance(node.func, ast.Name) and node.func.id == "next")
            ):
                return
    raise ValueError("CODE_GENERATION_BLOCKED: CSV readers consume headers but no data rows")


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

    def prepare_attempt_input(
        self,
        input_data: CodeAgentInput,
        state: ProblemState,
        previous_errors: tuple[str, ...],
    ) -> CodeAgentInput:
        if not previous_errors:
            return input_data
        feedback = [
            f"AUTOMATED_SOLVE_RETRY_FEEDBACK: {message[:400]}"
            for message in previous_errors[-2:]
        ]
        return input_data.model_copy(
            update={"user_guidance": [*input_data.user_guidance, *feedback]}
        )

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
                            indent=None
                        ),
                        model_digest=mathematical_model_digest(input_data.mathematical_model),
                        algorithm_plan_json=input_data.algorithm_plan.model_dump_json(indent=None),
                        result_schema_json=json.dumps(
                            GeneratedResultPayload.model_json_schema(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        input_manifest_json=json.dumps(
                            input_data.input_manifest, ensure_ascii=False, separators=(",", ":")
                        ),
                        user_guidance=json.dumps(input_data.user_guidance, ensure_ascii=False),
                    ),
                ),
            ],
            metadata={"agent": self.name, "prompt_version": prompt.version},
        )
        response = await provider.structured_generate(request, GeneratedProgramDraft)
        reject_hardcoded_results(response.parsed)
        if input_data.mathematical_model.data_bindings and any(
            item.get("path", "").lower().endswith(".csv")
            for item in input_data.input_manifest
        ):
            reject_header_only_csv_usage(response.parsed)
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
            is_mock=route.selected_provider == "mock",
        )
        return AgentExecution(
            output=program,
            response=response.response,
            prompt_version=prompt.version,
        )
