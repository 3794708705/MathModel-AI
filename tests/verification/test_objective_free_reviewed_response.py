"""Regression for objective-free models backed by independent scenario replays."""

from types import SimpleNamespace
from uuid import uuid4

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    IndependentVerificationReport,
    MetricSpec,
    ReviewedValidationEvidence,
    ScenarioSpec,
    VerificationPlan,
    VerificationRequirements,
    VerifiedMetric,
)
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.verification import (
    RedTeamReport,
    RobustnessConfig,
    SensitivityConfig,
    ValidationCheckCategory,
)
from mathmodel_ai.verification.metric_recompute import content_digest
from mathmodel_ai.verification.quality_gates import (
    robustness_quality_gate,
    sensitivity_quality_gate,
    verified_result_quality_gate,
)
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from tests.verification.helpers import experiment_engine, valid_report


def _evidence(model_digest, result_id):
    def metric(metric_id):
        return MetricSpec(
            metric_id=metric_id,
            key="algebraic_scalar",
            version="1",
            value_symbol="response",
            quantity="synthetic response",
            calculation="independent scalar recalculation",
            inputs=["response"],
            unit="dimensionless",
            tolerance_provenance="synthetic fixture tolerance",
            model_binding=model_digest,
        )

    baseline_metric = metric("baseline_response")
    perturbed_metric = metric("perturbed_response")
    scenarios = [
        ScenarioSpec(
            scenario_id="baseline",
            version="1",
            decision_values={"driver": 2.0},
            input_changes=["driver"],
            metrics=[baseline_metric],
            baseline="synthetic reference",
            perturbation="driver changed",
            reason="response comparison",
            comparison_quantity="response",
            acceptance_criterion="independent metric pass",
            criterion_provenance="synthetic fixture",
        ),
        ScenarioSpec(
            scenario_id="perturbed",
            version="1",
            parameter_values={"rate": 1.01},
            input_changes=["rate"],
            metrics=[perturbed_metric],
            baseline="synthetic reference",
            perturbation="rate changed",
            reason="parameter sensitivity",
            comparison_quantity="response",
            acceptance_criterion="independent metric pass",
            criterion_provenance="synthetic fixture",
        ),
    ]
    policy = VerificationRequirements(
        version="2",
        review_status="REVIEWED",
        production_eligible=True,
        benchmark_id="BENCH-synthetic",
        manifest_digest="a" * 64,
        model_digest=model_digest,
        metrics=[baseline_metric],
        scenarios=scenarios,
        scientific_scope="Synthetic objective-free response verification.",
    )
    plan = VerificationPlan(
        attempt_id=uuid4(),
        result_id=result_id,
        version="1",
        source_artifact_id=uuid4(),
        source_sha256="b" * 64,
        metrics=policy.metrics,
        scenarios=scenarios,
        scientific_scope=policy.scientific_scope,
    )
    replays = [
        SimpleNamespace(
            scenario_id=spec.scenario_id,
            status=IndependentStatus.PASS,
            execution=SimpleNamespace(run_id=uuid4(), is_mock=False),
            metrics=[
                VerifiedMetric(
                    metric_id=spec.metrics[0].metric_id,
                    key=spec.metrics[0].key,
                    status="PASS",
                    verified=float(index + 2),
                    source_digest="c" * 64,
                    calculator_digest="d" * 64,
                )
            ],
        )
        for index, spec in enumerate(scenarios)
    ]
    report = IndependentVerificationReport.model_construct(
        report_id=plan.plan_id,
        attempt_id=plan.attempt_id,
        plan_id=plan.plan_id,
        plan_digest=content_digest(plan),
        result_id=result_id,
        status=IndependentStatus.PASS,
        metrics=[replays[0].metrics[0]],
        replays=replays,
        required_metrics=1,
        passed_metrics=1,
        required_scenarios=2,
        passed_scenarios=2,
        errors=[],
    )
    return ReviewedValidationEvidence.model_construct(requirements=policy, plan=plan, report=report)


def test_objective_free_response_requires_exact_independent_evidence():
    model, result, validation, _ = valid_report()
    model = model.model_copy(update={"model_family": ModelFamily.DYNAMIC_SYSTEM, "objective": None})
    digest = mathematical_model_digest(model)
    result = result.model_copy(update={"objective": None, "model_digest": digest})
    validation = validation.model_copy(update={"model_digest": digest})
    evidence = _evidence(digest, result.result_id)
    engine = experiment_engine()
    sensitivity, executions = SensitivityAnalyzer(engine).analyze(
        model=model,
        result=result,
        validation=validation,
        config=SensitivityConfig(),
        reviewed_evidence=evidence,
    )
    assert executions == []
    assert sensitivity.baseline_objective is None
    assert set(sensitivity.reviewed_replay_ids) == {"perturbed"}
    assert sensitivity.reviewed_metric_values == {"perturbed_response": 3.0}
    assert sensitivity_quality_gate(sensitivity, evidence).status.value == "PASS"
    assert sensitivity_quality_gate(sensitivity).status.value != "PASS"

    robustness, executions = RobustnessAnalyzer(engine).analyze(
        model=model,
        result=result,
        validation=validation,
        sensitivity=sensitivity,
        config=RobustnessConfig(),
        reviewed_evidence=evidence,
    )
    assert executions == []
    assert robustness.baseline_objective is None
    assert set(robustness.reviewed_replay_ids) == {"baseline", "perturbed"}
    assert robustness_quality_gate(robustness, evidence).status.value == "PASS"
    assert robustness_quality_gate(robustness).status.value != "PASS"
    tampered = sensitivity.model_copy(
        update={"reviewed_metric_values": {"perturbed_response": 9.0}}
    )
    assert sensitivity_quality_gate(tampered, evidence).status.value != "PASS"

    # The final result gate still requires a non-Mock red-team review, exact
    # identity chaining, and the independently recomputed response summaries.
    validation = validation.model_copy(
        update={
            "metric_recalculations": [
                item.model_copy(update={"category": ValidationCheckCategory.OUTPUT})
                for item in validation.metric_recalculations
            ]
        }
    )
    red_team = RedTeamReport(
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=digest,
        result_id=result.result_id,
        validation_id=validation.validation_id,
        sensitivity_id=sensitivity.sensitivity_id,
        robustness_id=robustness.robustness_id,
        critical_count=0,
        major_count=0,
        minor_count=0,
        summary="Synthetic independent red-team review.",
        reviewer_agent_run_id=uuid4(),
        status="PASS",
    )
    gate = verified_result_quality_gate(
        validation,
        sensitivity,
        robustness,
        red_team,
        validation_integrity_errors=[],
        sensitivity_integrity_errors=[],
        robustness_integrity_errors=[],
        reviewed_evidence=evidence,
    )
    assert gate.status.value == "PASS", gate.errors
    tampered_gate = verified_result_quality_gate(
        validation,
        tampered,
        robustness,
        red_team,
        validation_integrity_errors=[],
        sensitivity_integrity_errors=[],
        robustness_integrity_errors=[],
        reviewed_evidence=evidence,
    )
    assert tampered_gate.status.value != "PASS"
