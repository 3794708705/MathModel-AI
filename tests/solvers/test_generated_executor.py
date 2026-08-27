from __future__ import annotations

from unittest.mock import Mock

import pytest

from mathmodel_ai.core.errors import DependencyUnavailableError, SandboxError
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import ExecutionOrigin
from mathmodel_ai.schemas.program import (
    GeneratedProgram,
    GeneratedProgramStatus,
    GeneratedSourceFile,
    generated_program_hash,
)
from mathmodel_ai.schemas.solver import SolverOptions
from mathmodel_ai.solvers.generated import GeneratedProgramExecutor
from tests.mathematical.helpers import lp_model


def _program(*, content: str, dependencies: list[str]) -> GeneratedProgram:
    model = lp_model()
    source = GeneratedSourceFile(path="solve.py", content=content).with_digest()
    return GeneratedProgram(
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=mathematical_model_digest(model),
        entrypoint="solve.py",
        files=[source],
        dependencies=dependencies,
        solver_target="SCIPY_HIGHS",
        explanation="generated executor validation fixture",
        code_hash=generated_program_hash([source]),
        generated_by="code_agent",
        prompt_version="fixture",
        execution_origin=ExecutionOrigin.GENERATED_PROGRAM,
        status=GeneratedProgramStatus.READY,
    )


def _executor(sandbox: Mock) -> GeneratedProgramExecutor:
    return GeneratedProgramExecutor(
        sandbox=sandbox,
        store=Mock(spec=FileStore),
    )


def test_generated_executor_rejects_unapproved_and_uninstalled_dependencies() -> None:
    sandbox = Mock(spec=SandboxExecutor)
    sandbox.probe_python_module.return_value = False
    executor = _executor(sandbox)
    program = _program(content="print('solver')", dependencies=["random-package"])
    model = lp_model(
        project_id=program.project_id,
        problem_id=program.problem_id,
        model_id=program.model_id,
    )

    with pytest.raises(DependencyUnavailableError, match="random-package"):
        executor.execute(program, model, SolverOptions())

    scipy_program = _program(content="print('solver')", dependencies=["scipy"])
    scipy_model = lp_model(
        project_id=scipy_program.project_id,
        problem_id=scipy_program.problem_id,
        model_id=scipy_program.model_id,
    )
    with pytest.raises(DependencyUnavailableError, match="scipy"):
        executor.execute(scipy_program, scipy_model, SolverOptions())


def test_generated_executor_rejects_dynamic_install_and_noncanonical_model_digest() -> None:
    sandbox = Mock(spec=SandboxExecutor)
    sandbox.probe_python_module.return_value = True
    executor = _executor(sandbox)
    dynamic = _program(
        content="import subprocess\nsubprocess.run(['pip', 'install', 'thing'])",
        dependencies=[],
    )
    model = lp_model(
        project_id=dynamic.project_id,
        problem_id=dynamic.problem_id,
        model_id=dynamic.model_id,
    )

    with pytest.raises(DependencyUnavailableError, match="dynamic installation"):
        executor.execute(dynamic, model, SolverOptions())

    wrong_digest = _program(content="print('solver')", dependencies=[]).model_copy(
        update={"model_digest": "f" * 64}
    )
    matching_model = lp_model(
        project_id=wrong_digest.project_id,
        problem_id=wrong_digest.problem_id,
        model_id=wrong_digest.model_id,
    )
    with pytest.raises(SandboxError, match="model_digest is not canonical"):
        executor.execute(wrong_digest, matching_model, SolverOptions())
