import subprocess
import sys


def test_benchmark_schema_import_does_not_depend_on_prior_provider_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mathmodel_ai.benchmark.evaluation import BenchmarkEvaluator; "
            "assert BenchmarkEvaluator is not None",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        shell=False,
    )

    assert completed.returncode == 0, completed.stderr
