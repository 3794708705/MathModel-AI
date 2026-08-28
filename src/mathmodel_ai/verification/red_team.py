from __future__ import annotations

from uuid import UUID

from mathmodel_ai.schemas.mathematical import MathematicalModel, UnitCheckStatus
from mathmodel_ai.schemas.verification import (
    ExperimentReportStatus,
    RedTeamCategory,
    RedTeamDraft,
    RedTeamFinding,
    RedTeamReport,
    RedTeamSeverity,
    RobustnessReport,
    SensitivityReport,
    ValidationReport,
    ValidationStatus,
)


class RedTeamAnalyzer:
    def compile(
        self,
        *,
        model: MathematicalModel,
        validation: ValidationReport,
        sensitivity: SensitivityReport,
        robustness: RobustnessReport,
        draft: RedTeamDraft,
        reviewer_agent_run_id: UUID,
        review_is_mock: bool,
    ) -> RedTeamReport:
        deterministic = self._deterministic_findings(
            model=model,
            validation=validation,
            sensitivity=sensitivity,
            robustness=robustness,
        )
        findings: list[RedTeamFinding] = []
        identifiers: set[str] = set()
        for item in [*deterministic, *draft.findings]:
            if item.finding_id in identifiers:
                suffix = 2
                candidate = f"{item.finding_id}-{suffix}"
                while candidate in identifiers:
                    suffix += 1
                    candidate = f"{item.finding_id}-{suffix}"
                item = item.model_copy(update={"finding_id": candidate})
            identifiers.add(item.finding_id)
            findings.append(item)
        critical = sum(
            item.severity is RedTeamSeverity.CRITICAL and not item.resolved for item in findings
        )
        major = sum(
            item.severity is RedTeamSeverity.MAJOR and not item.resolved for item in findings
        )
        minor = sum(
            item.severity is RedTeamSeverity.MINOR and not item.resolved for item in findings
        )
        status = (
            ValidationStatus.FAIL
            if critical
            else ValidationStatus.INCONCLUSIVE
            if review_is_mock
            else ValidationStatus.PASS
        )
        return RedTeamReport(
            project_id=model.project_id,
            problem_id=model.problem_id,
            model_id=model.model_id,
            model_version=model.version,
            model_digest=validation.model_digest,
            result_id=validation.result_id,
            validation_id=validation.validation_id,
            sensitivity_id=sensitivity.sensitivity_id,
            robustness_id=robustness.robustness_id,
            findings=findings,
            critical_count=critical,
            major_count=major,
            minor_count=minor,
            summary=draft.summary,
            residual_risks=list(dict.fromkeys(draft.residual_risks)),
            reviewer_agent_run_id=reviewer_agent_run_id,
            review_is_mock=review_is_mock,
            status=status,
        )

    @staticmethod
    def _deterministic_findings(
        *,
        model: MathematicalModel,
        validation: ValidationReport,
        sensitivity: SensitivityReport,
        robustness: RobustnessReport,
    ) -> list[RedTeamFinding]:
        findings: list[RedTeamFinding] = []
        if validation.status is not ValidationStatus.PASS:
            findings.append(
                RedTeamFinding(
                    finding_id="RTF-DETERMINISTIC-VALIDATION",
                    severity=RedTeamSeverity.CRITICAL,
                    category=RedTeamCategory.RESULT_INTERPRETATION,
                    title="Independent validation did not pass",
                    attack=(
                        "Treating this result as reliable would bypass failed or incomplete checks."
                    ),
                    evidence_refs=[f"validation:{validation.validation_id}"],
                    affected_refs=[f"result:{validation.result_id}"],
                    recommendation=(
                        "Correct the model/result chain and rerun independent validation."
                    ),
                    deterministic=True,
                )
            )
        if sensitivity.status is not ExperimentReportStatus.PASS:
            findings.append(
                RedTeamFinding(
                    finding_id="RTF-DETERMINISTIC-SENSITIVITY",
                    severity=RedTeamSeverity.MAJOR,
                    category=RedTeamCategory.SENSITIVITY,
                    title="Sensitivity evidence is incomplete",
                    attack=(
                        "Parameter conclusions may depend on failed or missing perturbation runs."
                    ),
                    evidence_refs=[f"sensitivity:{sensitivity.sensitivity_id}"],
                    recommendation="Resolve failed perturbations or narrow the supported claim.",
                    deterministic=True,
                )
            )
        if robustness.status is not ExperimentReportStatus.PASS:
            findings.append(
                RedTeamFinding(
                    finding_id="RTF-DETERMINISTIC-ROBUSTNESS",
                    severity=RedTeamSeverity.CRITICAL,
                    category=RedTeamCategory.ROBUSTNESS,
                    title="Robustness scenarios are not uniformly feasible",
                    attack="The proposed conclusion fails under one or more declared scenarios.",
                    evidence_refs=[f"robustness:{robustness.robustness_id}"],
                    recommendation=(
                        "Repair the formulation or explicitly constrain the valid scenario range."
                    ),
                    deterministic=True,
                )
            )
        if sensitivity.maximum_absolute_relative_change is not None and (
            sensitivity.maximum_absolute_relative_change > 1
        ):
            findings.append(
                RedTeamFinding(
                    finding_id="RTF-DETERMINISTIC-HIGH-SENSITIVITY",
                    severity=RedTeamSeverity.MAJOR,
                    category=RedTeamCategory.PARAMETER,
                    title="Objective is highly sensitive to tested parameters",
                    attack=(
                        "A competition conclusion may reverse under an allowed parameter "
                        "perturbation."
                    ),
                    evidence_refs=[f"sensitivity:{sensitivity.sensitivity_id}"],
                    recommendation=(
                        "Report the sensitive parameter and justify or estimate it more carefully."
                    ),
                    deterministic=True,
                )
            )
        unknown_units = [
            item.equation_id
            for item in model.equations
            if item.dimension_status is UnitCheckStatus.UNKNOWN
        ]
        if unknown_units:
            findings.append(
                RedTeamFinding(
                    finding_id="RTF-DETERMINISTIC-UNITS",
                    severity=RedTeamSeverity.MAJOR,
                    category=RedTeamCategory.MODEL_STRUCTURE,
                    title="Some equation dimensions remain unknown",
                    attack="An unresolved unit token can hide a scale or dimensional error.",
                    evidence_refs=[f"model:{model.model_id}:v{model.version}"],
                    affected_refs=unknown_units,
                    recommendation=(
                        "Resolve every custom unit before relying on scaled conclusions."
                    ),
                    deterministic=True,
                )
            )
        return findings
