from __future__ import annotations

import json

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.execution import ExecutionOrigin
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.program import (
    GeneratedProgram,
    GeneratedProgramStatus,
    GeneratedSourceFile,
    generated_program_hash,
)
from mathmodel_ai.schemas.solver import SolverFamily, SolverOptions


def build_deterministic_program(
    model: MathematicalModel,
    options: SolverOptions,
    *,
    backend: str,
    target: SolverFamily,
    dependencies: list[str],
) -> GeneratedProgram:
    payload = json.dumps(
        {
            "backend": backend,
            "model": model.model_dump(mode="json"),
            "options": options.model_dump(mode="json"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    source = GeneratedSourceFile(
        path="main.py",
        content=(
            "import json\n"
            "from pathlib import Path\n"
            "from mathmodel_solver_runtime import solve\n\n"
            f"PAYLOAD = json.loads({payload!r})\n"
            "RESULT = solve(PAYLOAD)\n"
            "Path('/output/result.json').write_text(\n"
            "    json.dumps(RESULT, ensure_ascii=False, allow_nan=False),\n"
            "    encoding='utf-8',\n"
            ")\n"
        ),
    ).with_digest()
    return GeneratedProgram(
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=mathematical_model_digest(model),
        language="PYTHON",
        entrypoint="main.py",
        files=[source],
        dependencies=dependencies,
        solver_target=target.value,
        explanation=(
            "Deterministic adapter payload executed by the versioned solver runtime; "
            "no LLM-generated solver boilerplate or numeric result is embedded."
        ),
        code_hash=generated_program_hash([source]),
        generated_by=f"deterministic:{backend}",
        prompt_version="deterministic-translator-v1",
        execution_origin=ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER,
        status=GeneratedProgramStatus.READY,
        is_mock=False,
    )
