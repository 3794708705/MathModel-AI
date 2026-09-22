import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    frontend_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="mathmodel-openapi-") as temporary_root:
        root = Path(temporary_root)
        os.environ["MM_ENVIRONMENT"] = "test"
        os.environ["MM_DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
        os.environ["MM_STORAGE_ROOT"] = str(root / "storage")
        os.environ["MM_SANDBOX_ROOT"] = str(root / "sandbox")
        os.environ["MM_SOLVER_SANDBOX_ROOT"] = str(root / "solver")
        os.environ["MM_BENCHMARK_CACHE_ROOT"] = str(root / "benchmark-cache")
        os.environ["MM_BENCHMARK_ARTIFACT_ROOT"] = str(root / "benchmark-runs")

        from mathmodel_ai.main import create_app

        app = create_app()
        output = frontend_root / "openapi.json"
        output.write_text(
            json.dumps(app.openapi(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
