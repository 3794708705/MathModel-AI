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
            readers.update(target.id for target in targets if isinstance(target, ast.Name))
    if not readers:
        return

    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and any(
                isinstance(item, ast.Name) and item.id in readers for item in ast.walk(node.iter)
            ):
                return
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {
                    "read_csv",
                    "scan_csv",
                }
                and not any(
                    keyword.arg == "nrows"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == 0
                    for keyword in node.keywords
                )
            ):
                return
            if any(isinstance(arg, ast.Name) and arg.id in readers for arg in node.args) and not (
                isinstance(node.func, ast.Name) and node.func.id == "next"
            ):
                return
    raise ValueError("CODE_GENERATION_BLOCKED: CSV readers consume headers but no data rows")


def reject_discarded_csv_rows(program: GeneratedProgramDraft) -> None:
    """Reject the observed no-op row loop used only to make a CSV appear consumed."""
    readers: set[str] = set()
    trees = [ast.parse(source.content, filename=source.path) for source in program.files]
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not isinstance(call.func, ast.Attribute) or call.func.attr not in {
                "reader",
                "DictReader",
            }:
                continue
            readers.update(target.id for target in node.targets if isinstance(target, ast.Name))
    if not readers:
        return
    row_loops: list[tuple[str, list[ast.stmt]]] = []
    for tree in trees:
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.For, ast.AsyncFor))
                and isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Name)
                and node.iter.id in readers
            ):
                row_loops.append((node.target.id, node.body))
    if row_loops and not any(
        any(
            isinstance(item, ast.Name) and item.id == row_name and isinstance(item.ctx, ast.Load)
            for statement in body
            for item in ast.walk(statement)
        )
        for row_name, body in row_loops
    ):
        raise ValueError("CODE_GENERATION_BLOCKED: CSV rows are iterated but their values unused")


def reject_sorted_groupby_in_seeded_protocol(program: GeneratedProgramDraft) -> None:
    """A seeded sequential shuffle must not silently reorder its strata."""
    for source in program.files:
        if source.path != program.entrypoint:
            continue
        try:
            tree = ast.parse(source.content, filename=source.path)
        except SyntaxError as exc:
            raise ValueError(f"CODE_GENERATION_BLOCKED: invalid Python in {source.path}") from exc
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "groupby"
            ):
                continue
            preserves_csv_order = any(
                keyword.arg == "sort"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in node.keywords
            )
            if not preserves_csv_order:
                raise ValueError(
                    "CODE_GENERATION_BLOCKED: seeded conditional permutation "
                    "requires groupby(sort=False) to preserve first-seen stratum order"
                )


def reject_reused_strata_in_seeded_protocol(program: GeneratedProgramDraft) -> None:
    """Catch in-place shuffles of persistent strata across repeated null draws."""
    entrypoint = next(item for item in program.files if item.path == program.entrypoint)
    tree = ast.parse(entrypoint.content, filename=entrypoint.path)
    for loop in ast.walk(tree):
        if not (
            isinstance(loop, ast.For)
            and isinstance(loop.target, ast.Name)
            and loop.target.id == "_"
            and isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Name)
            and loop.iter.func.id == "range"
        ):
            continue
        for node in ast.walk(ast.Module(body=loop.body, type_ignores=[])):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "shuffle"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Subscript)
                and isinstance(node.args[0].value, ast.Name)
            ):
                continue
            container = node.args[0].value.id
            refreshed = any(
                isinstance(item, (ast.Assign, ast.AnnAssign))
                and any(
                    (isinstance(target, ast.Name) and target.id == container)
                    or (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == container
                    )
                    for target in (item.targets if isinstance(item, ast.Assign) else [item.target])
                )
                for item in ast.walk(ast.Module(body=loop.body, type_ignores=[]))
            )
            if not refreshed:
                raise ValueError(
                    "CODE_GENERATION_BLOCKED: seeded conditional permutation "
                    "reuses in-place shuffled strata across null draws"
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

    def prepare_attempt_input(
        self,
        input_data: CodeAgentInput,
        state: ProblemState,
        previous_errors: tuple[str, ...],
    ) -> CodeAgentInput:
        if not previous_errors:
            return input_data
        feedback = [
            f"AUTOMATED_SOLVE_RETRY_FEEDBACK: {message[:400]}" for message in previous_errors[-2:]
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
        if any("causal_science:randomness_test" in item for item in input_data.user_guidance):
            reject_sorted_groupby_in_seeded_protocol(response.parsed)
            reject_reused_strata_in_seeded_protocol(response.parsed)
        if input_data.mathematical_model.data_bindings and any(
            item.get("path", "").lower().endswith(".csv") for item in input_data.input_manifest
        ):
            reject_header_only_csv_usage(response.parsed)
            reject_discarded_csv_rows(response.parsed)
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
