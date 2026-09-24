"""Objective-free response evidence from independently audited scenario replays.

This is a separate contract from objective perturbation experiments: a dynamic
model has no objective delta, so none is invented. The independent verifier
has already re-executed and audited each reviewed scenario before this adapter
can be used; the adapter additionally checks the complete in-memory binding.
"""

from uuid import UUID

from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    ReviewedValidationEvidence,
)
from mathmodel_ai.verification.metric_recompute import content_digest


def reviewed_response_summary(
    evidence: ReviewedValidationEvidence,
    *,
    model_digest: str,
    result_id: UUID,
    parameter_only: bool,
) -> tuple[dict[str, UUID], dict[str, float]]:
    policy, plan, report = evidence.requirements, evidence.plan, evidence.report
    if (
        policy.model_digest != model_digest
        or plan.result_id != result_id
        or report.result_id != result_id
        or plan.attempt_id != report.attempt_id
        or report.plan_id != plan.plan_id
        or report.report_id != plan.plan_id
        or report.plan_digest != content_digest(plan)
        or plan.scenarios != policy.scenarios
        or plan.metrics != policy.metrics
        or plan.scientific_scope != policy.scientific_scope
        or not policy.matches_observation(plan)
        or report.status is not IndependentStatus.PASS
        or report.errors
        or report.required_scenarios != sum(s.required for s in plan.scenarios)
        or report.passed_scenarios != report.required_scenarios
        or report.required_metrics != sum(m.required for m in plan.metrics)
        or report.passed_metrics != report.required_metrics
        or len(report.replays) != len(plan.scenarios)
    ):
        raise ValueError("reviewed response evidence is not bound to the passing formal result")
    replays = {item.scenario_id: item for item in report.replays}
    if len(replays) != len(report.replays) or set(replays) != {
        item.scenario_id for item in plan.scenarios
    }:
        raise ValueError("reviewed response replay coverage is incomplete")
    if len({item.execution.run_id for item in report.replays}) != len(report.replays):
        raise ValueError("reviewed response reuses a scenario execution")
    selected = [
        spec
        for spec in plan.scenarios
        if spec.required and (not parameter_only or bool(spec.parameter_values))
    ]
    if not selected or (
        not parameter_only
        and not any(spec.parameter_values or spec.decision_values for spec in selected)
    ):
        raise ValueError("reviewed response lacks the required perturbation coverage")
    baseline_metrics = {item.metric_id: item for item in report.metrics}
    if len(baseline_metrics) != len(report.metrics) or set(baseline_metrics) != {
        item.metric_id for item in plan.metrics
    }:
        raise ValueError("reviewed baseline metric coverage is incomplete")
    if any(
        baseline_metrics[spec.metric_id].status is not IndependentStatus.PASS
        or baseline_metrics[spec.metric_id].verified is None
        for spec in plan.metrics
        if spec.required
    ):
        raise ValueError("reviewed baseline metric did not pass")
    replay_ids: dict[str, UUID] = {}
    values: dict[str, float] = {}
    for spec in plan.scenarios:
        replay = replays[spec.scenario_id]
        metric_by_id = {item.metric_id: item for item in replay.metrics}
        if (
            replay.status is not IndependentStatus.PASS
            or replay.execution.is_mock
            or len(metric_by_id) != len(replay.metrics)
            or set(metric_by_id) != {item.metric_id for item in spec.metrics}
        ):
            raise ValueError(f"reviewed response replay failed: {spec.scenario_id}")
        for metric in spec.metrics:
            actual = metric_by_id[metric.metric_id]
            if metric.required and (
                actual.status is not IndependentStatus.PASS or actual.verified is None
            ):
                raise ValueError(f"reviewed response metric failed: {metric.metric_id}")
            if spec in selected and metric.required:
                assert actual.verified is not None
                values[metric.metric_id] = actual.verified
        if spec in selected:
            replay_ids[spec.scenario_id] = replay.execution.run_id
    if not values:
        raise ValueError("reviewed response has no independently verified numeric outputs")
    return replay_ids, values
