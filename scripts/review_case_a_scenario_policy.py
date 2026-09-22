"""Execute every reviewed Case A metric/scenario before policy use."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import SandboxLimits
from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    RawMetricOutput,
    VerificationRequirements,
)
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.solvers.router import SolverRouter
from mathmodel_ai.solvers.scipy import SciPySolver
from mathmodel_ai.verification.metric_recompute import content_digest, recompute
from mathmodel_ai.verification.replay_integrity import audit_replay
from mathmodel_ai.verification.scenario_replay import ScenarioReplayer

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "case-003-mcm-2024-a"
REPORT = CASE / "verification-policy-replay-review.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def main() -> None:
    contract = _load(CASE / "mathematical-model-v2.json")
    policy_payload = _load(CASE / "independent-verification.json")
    model = MathematicalModel.model_validate(contract["model"])
    policy = VerificationRequirements.model_validate(policy_payload)
    model_digest = mathematical_model_digest(model)
    policy_digest = content_digest(policy)
    if model_digest != policy.model_digest or model_digest != contract["model_digest"]:
        raise ValueError("review inputs do not share the mathematical model digest")

    results: list[dict[str, Any]] = []
    execution_ids = []
    with tempfile.TemporaryDirectory(prefix="mathmodel-case-a-policy-review-") as name:
        root = Path(name)
        store = LocalFileStore(root / "store")
        limits = SandboxLimits(
            cpu_cores=1,
            memory_mb=512,
            timeout_seconds=30,
            pids_limit=64,
            max_output_bytes=1_000_000,
            max_artifacts=4,
            max_artifact_bytes=16 * 1024 * 1024,
        )
        sandbox = SandboxExecutor(
            store=store,
            root=root / "sandbox",
            image="mathmodel-ai-solver:phase4",
            limits=limits,
        )
        replayer = ScenarioReplayer(
            store=store,
            root=root / "replays",
            image="mathmodel-ai-solver:phase4",
            solver_router=SolverRouter([SciPySolver(sandbox=sandbox, store=store)]),
        )
        for scenario in policy.scenarios:
            replay = replayer.execute(model, scenario)
            errors = audit_replay(model, scenario, replay, store)
            execution_ids.append(replay.execution.run_id)
            results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_digest": replay.scenario_digest,
                    "status": replay.status.value,
                    "execution_status": replay.execution.status.value,
                    "exit_code": replay.execution.exit_code,
                    "code_hash": replay.execution.code_hash,
                    "output_digest": replay.output_digest,
                    "required_metrics": sum(item.required for item in scenario.metrics),
                    "passed_metrics": sum(
                        item.status is IndependentStatus.PASS
                        for item in replay.metrics
                        if item.metric_id
                        in {spec.metric_id for spec in scenario.metrics if spec.required}
                    ),
                    "audit_errors": errors,
                }
            )

    metric_results = [
        recompute(
            metric,
            RawMetricOutput(variables={}),
            source_digest=policy_digest,
            model=model,
        )
        for metric in policy.metrics
    ]
    required_scenarios = [item for item in policy.scenarios if item.required]
    critical_count = sum(
        item["status"] != IndependentStatus.PASS.value or bool(item["audit_errors"])
        for item in results
        if item["scenario_id"] in {scenario.scenario_id for scenario in required_scenarios}
    ) + sum(item.status is not IndependentStatus.PASS for item in metric_results)
    if len(execution_ids) != len(set(execution_ids)):
        critical_count += 1
    report: dict[str, Any] = {
        "review_version": "1",
        "benchmark_id": policy.benchmark_id,
        "model_digest": model_digest,
        "model_contract_digest": contract["content_digest"],
        "policy_digest": policy_digest,
        "reviewer": "deterministic-real-docker-scenario-review-v1",
        "reviewer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_source_sha256": hashlib.sha256(
            (ROOT / "src/mathmodel_ai/verification/replay_runtime.py").read_bytes()
        ).hexdigest(),
        "scenario_replay_source_sha256": hashlib.sha256(
            (ROOT / "src/mathmodel_ai/verification/scenario_replay.py").read_bytes()
        ).hexdigest(),
        "required_metrics": sum(item.required for item in policy.metrics),
        "passed_metrics": sum(
            item.status is IndependentStatus.PASS
            for item in metric_results
            if item.metric_id in {spec.metric_id for spec in policy.metrics if spec.required}
        ),
        "required_scenarios": len(required_scenarios),
        "passed_scenarios": sum(
            item["status"] == IndependentStatus.PASS.value and not item["audit_errors"]
            for item in results
            if item["scenario_id"] in {scenario.scenario_id for scenario in required_scenarios}
        ),
        "unique_execution_count": len(set(execution_ids)),
        "scenario_results": results,
        "critical_count": critical_count,
        "status": "PASS" if critical_count == 0 else "FAIL",
        "content_digest": None,
    }
    report["content_digest"] = content_digest(
        {key: value for key, value in report.items() if key != "content_digest"}
    )
    if critical_count:
        raise ValueError("reviewed policy has failing required metrics or scenarios")
    REPORT.write_bytes(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    )
    print(
        f"PASS metrics={report['passed_metrics']}/{report['required_metrics']} "
        f"scenarios={report['passed_scenarios']}/{report['required_scenarios']}"
    )


if __name__ == "__main__":
    main()
