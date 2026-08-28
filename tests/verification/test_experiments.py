import pytest
from pydantic import ValidationError

from mathmodel_ai.schemas.verification import (
    ExperimentReportStatus,
    RobustnessConfig,
    RobustnessMethod,
    SensitivityConfig,
)
from mathmodel_ai.verification.quality_gates import (
    robustness_quality_gate,
    sensitivity_quality_gate,
)
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from tests.verification.helpers import experiment_engine, valid_report


def test_sensitivity_executes_default_signed_perturbations_and_summarizes() -> None:
    model, result, validation, _ = valid_report()
    analyzer = SensitivityAnalyzer(experiment_engine())

    report, outcomes = analyzer.analyze(
        model=model,
        result=result,
        validation=validation,
        config=SensitivityConfig(perturbation_fractions=[0.05, 0.10]),
    )

    assert report.status is ExperimentReportStatus.PASS
    assert len(outcomes) == 4
    assert {item.record.perturbations[0].fraction for item in outcomes} == {
        -0.10,
        -0.05,
        0.05,
        0.10,
    }
    assert report.objective_min == pytest.approx(27)
    assert report.objective_max == pytest.approx(33)
    assert report.parameter_elasticities["demand"] == pytest.approx(1)
    assert all(item.record.execution_record_id is not None for item in outcomes)
    assert sensitivity_quality_gate(report).status.value == "PASS"


def test_robustness_is_seeded_and_bootstrap_blocks_without_data_contract() -> None:
    model, result, validation, _ = valid_report()
    engine = experiment_engine()
    sensitivity, _ = SensitivityAnalyzer(engine).analyze(
        model=model,
        result=result,
        validation=validation,
        config=SensitivityConfig(perturbation_fractions=[0.05]),
    )
    analyzer = RobustnessAnalyzer(engine)
    config = RobustnessConfig(
        method=RobustnessMethod.MONTE_CARLO,
        sample_count=5,
        random_seed=7,
    )

    first, _ = analyzer.analyze(
        model=model,
        result=result,
        validation=validation,
        sensitivity=sensitivity,
        config=config,
    )
    second, _ = analyzer.analyze(
        model=model,
        result=result,
        validation=validation,
        sensitivity=sensitivity,
        config=config,
    )
    blocked, _ = analyzer.analyze(
        model=model,
        result=result,
        validation=validation,
        sensitivity=sensitivity,
        config=RobustnessConfig(method=RobustnessMethod.BOOTSTRAP),
    )

    assert first.status is ExperimentReportStatus.PASS
    assert [item.perturbations for item in first.experiments] == [
        item.perturbations for item in second.experiments
    ]
    assert first.summary.feasibility_rate == 1
    assert robustness_quality_gate(first).status.value == "PASS"
    assert blocked.status is ExperimentReportStatus.BLOCKED
    assert robustness_quality_gate(blocked).status.value == "HUMAN_REVIEW"


def test_experiment_config_rejects_unsafe_or_ambiguous_ranges() -> None:
    with pytest.raises(ValidationError):
        SensitivityConfig(perturbation_fractions=[0.1, 0.1])
    with pytest.raises(ValidationError):
        SensitivityConfig(perturbation_fractions=[1.1])
    with pytest.raises(ValidationError):
        RobustnessConfig(scenario_fractions=[-0.1, -0.1])
