from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from mathmodel_ai.mathematical.repository import MathematicalRepository, ResultContext
from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.schemas.execution import ExecutionStatus
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.paper import (
    EvidenceProvenance,
    EvidenceRecord,
    EvidenceSnapshot,
    EvidenceType,
    EvidenceVerificationStatus,
    ReferenceMetadataStatus,
    ReferenceRecord,
)
from mathmodel_ai.schemas.problem_analysis import EvidenceStatus
from mathmodel_ai.schemas.problem_state import ProblemState, WorkflowStatus
from mathmodel_ai.schemas.quality import QualityGateStatus
from mathmodel_ai.schemas.solver import SolverStatus
from mathmodel_ai.schemas.verification import (
    ExperimentReportStatus,
    RedTeamReport,
    RedTeamSeverity,
    RobustnessReport,
    SensitivityReport,
    ValidationReport,
    ValidationStatus,
)
from mathmodel_ai.verification.repository import VerificationRepository


class EvidenceBuildError(ValueError):
    """The selected result cannot provide a verified factual paper snapshot."""


def calculate_snapshot_hash(snapshot: EvidenceSnapshot, records: tuple[EvidenceRecord, ...]) -> str:
    """Recalculate an evidence snapshot from immutable leaf records."""

    return sha256_json(
        {
            "model_id": snapshot.verified_model_id,
            "model_version": snapshot.verified_model_version,
            "model_digest": snapshot.verified_model_digest,
            "result_id": snapshot.verified_result_id,
            "validation_id": snapshot.validation_id,
            "sensitivity_id": snapshot.sensitivity_id,
            "robustness_id": snapshot.robustness_id,
            "red_team_id": snapshot.red_team_report_id,
            "citations": snapshot.citation_reference_ids,
            "evidence": [
                {"id": item.evidence_id, "source_hash": item.provenance.source_hash}
                for item in records
            ],
        }
    )


@dataclass(frozen=True)
class VerifiedEvidenceBundle:
    state: ProblemState
    context: ResultContext
    validation: ValidationReport
    sensitivity: SensitivityReport
    robustness: RobustnessReport
    red_team: RedTeamReport
    records: tuple[EvidenceRecord, ...]
    snapshot: EvidenceSnapshot


class VerifiedEvidenceBuilder:
    """Build paper evidence only from the explicit Phase 5 verified result pointer."""

    def __init__(
        self,
        *,
        mathematical_repository: MathematicalRepository,
        verification_repository: VerificationRepository,
    ) -> None:
        self._mathematical = mathematical_repository
        self._verification = verification_repository

    def build(self, project_id: UUID) -> VerifiedEvidenceBundle:
        state = self._verification.load_current(project_id)
        result_id = state.verified_result_id
        if result_id is None:
            raise EvidenceBuildError("paper build requires explicit verified_result_id")
        self._require_verified_gate(state, result_id)
        context = self._mathematical.get_result_context(project_id, result_id)
        self._require_context(state, context)
        validation, sensitivity, robustness, red_team = self._resolve_unique_chain(
            state,
            result_id,
            context.model,
        )
        records = tuple(
            self._records(
                state,
                context,
                validation,
                sensitivity,
                robustness,
                red_team,
            )
        )
        snapshot = self._snapshot(
            context.model,
            result_id,
            validation,
            sensitivity,
            robustness,
            red_team,
            records,
        )
        return VerifiedEvidenceBundle(
            state=state,
            context=context,
            validation=validation,
            sensitivity=sensitivity,
            robustness=robustness,
            red_team=red_team,
            records=records,
            snapshot=snapshot,
        )

    def attach_references(
        self,
        bundle: VerifiedEvidenceBundle,
        references: list[ReferenceRecord],
    ) -> VerifiedEvidenceBundle:
        if any(
            item.project_id != bundle.state.project_id
            or item.metadata_status is not ReferenceMetadataStatus.VERIFIED
            for item in references
        ):
            raise EvidenceBuildError("only independently verified project references may attach")
        result = bundle.context.result
        common = {
            "model_id": result.model_id,
            "model_version": result.model_version,
            "model_digest": result.model_digest,
            "result_id": result.result_id,
            "solver_run_id": result.solver_run_id,
            "execution_record_id": result.execution_record_id,
            "validation_id": bundle.validation.validation_id,
            "sensitivity_id": bundle.sensitivity.sensitivity_id,
            "robustness_id": bundle.robustness.robustness_id,
            "red_team_report_id": bundle.red_team.report_id,
            "is_mock": False,
        }
        literature = tuple(
            self._record(
                bundle.state,
                EvidenceType.LITERATURE,
                "literature_reference",
                reference.reference_id,
                "1",
                reference.title,
                reference.model_dump(mode="json"),
                [
                    f"literature_source:{reference.source.value}:{reference.source_id}",
                    *([f"doi:{reference.doi}"] if reference.doi else []),
                ],
                common,
            )
            for reference in references
        )
        records = (*bundle.records, *literature)
        citation_ids = [item.reference_id for item in references]
        snapshot = bundle.snapshot.model_copy(
            update={
                "citation_reference_ids": citation_ids,
                "evidence_ids": [item.evidence_id for item in records],
                "snapshot_hash": "0" * 64,
            }
        )
        snapshot = snapshot.model_copy(
            update={"snapshot_hash": calculate_snapshot_hash(snapshot, records)}
        )
        return VerifiedEvidenceBundle(
            state=bundle.state,
            context=bundle.context,
            validation=bundle.validation,
            sensitivity=bundle.sensitivity,
            robustness=bundle.robustness,
            red_team=bundle.red_team,
            records=records,
            snapshot=snapshot,
        )

    @staticmethod
    def _require_verified_gate(state: ProblemState, result_id: UUID) -> None:
        gates = [
            item
            for item in state.quality_gates
            if item.gate == "VERIFIED" and item.subject_ref == f"result:{result_id}"
        ]
        if len(gates) != 1:
            raise EvidenceBuildError("verified result requires exactly one persisted VERIFIED gate")
        gate = gates[0]
        if (
            gate.status is not QualityGateStatus.PASS
            or gate.errors
            or not gate.checks
            or not all(gate.checks.values())
        ):
            raise EvidenceBuildError("persisted VERIFIED gate does not independently pass")
        if state.status is not WorkflowStatus.SUCCEEDED:
            raise EvidenceBuildError("paper evidence requires a successfully completed state")

    @staticmethod
    def _require_context(state: ProblemState, context: ResultContext) -> None:
        model = context.model
        result = context.result
        solver = context.solver_run
        execution = context.execution
        if not context.evidence.valid:
            raise EvidenceBuildError("result evidence chain failed deterministic verification")
        if result.result_id != state.verified_result_id:
            raise EvidenceBuildError("repository returned a result other than verified_result_id")
        identities = {
            (model.model_id, model.version),
            (result.model_id, result.model_version),
            (solver.model_id, solver.model_version),
        }
        digests = {context.evidence.model_digest_match, result.model_digest == solver.model_digest}
        if len(identities) != 1 or digests != {True}:
            raise EvidenceBuildError("model/result/solver revision identity does not match")
        if result.solver_run_id != solver.solver_run_id:
            raise EvidenceBuildError("result and solver run identity does not match")
        if (
            result.execution_record_id != execution.run_id
            or solver.execution_ref != execution.run_id
        ):
            raise EvidenceBuildError("result chain references a different execution")
        if (
            execution.is_mock
            or execution.status is not ExecutionStatus.SUCCEEDED
            or execution.exit_code != 0
            or not execution.network_disabled
            or not execution.non_root
        ):
            raise EvidenceBuildError("formal paper evidence requires a real safe execution")

    def _resolve_unique_chain(
        self,
        state: ProblemState,
        result_id: UUID,
        model: MathematicalModel,
    ) -> tuple[ValidationReport, SensitivityReport, RobustnessReport, RedTeamReport]:
        candidates: list[
            tuple[ValidationReport, SensitivityReport, RobustnessReport, RedTeamReport]
        ] = []
        for reference in state.red_team_reports:
            if (
                reference.model_id != model.model_id
                or reference.model_version != model.version
                or reference.model_digest
                != next(
                    item.model_digest
                    for item in state.result_records
                    if item.result_id == result_id
                )
                or reference.status is not ValidationStatus.PASS
                or reference.review_is_mock
            ):
                continue
            red_team = self._verification.get_red_team(state.project_id, reference.report_id)
            if red_team.result_id != result_id:
                continue
            robustness = self._verification.get_robustness(state.project_id, red_team.robustness_id)
            sensitivity = self._verification.get_sensitivity(
                state.project_id, red_team.sensitivity_id
            )
            validation = self._verification.get_validation(state.project_id, red_team.validation_id)
            if self._chain_passes(
                model,
                result_id,
                validation,
                sensitivity,
                robustness,
                red_team,
            ):
                candidates.append((validation, sensitivity, robustness, red_team))
        if len(candidates) != 1:
            raise EvidenceBuildError(
                "verified result must resolve to exactly one full Phase 5 verification chain"
            )
        return candidates[0]

    @staticmethod
    def _chain_passes(
        model: MathematicalModel,
        result_id: UUID,
        validation: ValidationReport,
        sensitivity: SensitivityReport,
        robustness: RobustnessReport,
        red_team: RedTeamReport,
    ) -> bool:
        model_keys = {
            (item.model_id, item.model_version, item.model_digest)
            for item in (validation, sensitivity, robustness, red_team)
        }
        expected_key = (model.model_id, model.version)
        return (
            len(model_keys) == 1
            and next(iter(model_keys))[:2] == expected_key
            and {item.result_id for item in (validation, sensitivity, robustness, red_team)}
            == {result_id}
            and validation.status is ValidationStatus.PASS
            and validation.evidence.valid
            and not validation.errors
            and sensitivity.status is ExperimentReportStatus.PASS
            and robustness.status is ExperimentReportStatus.PASS
            and red_team.status is ValidationStatus.PASS
            and not red_team.review_is_mock
            and not any(
                finding.severity is RedTeamSeverity.CRITICAL and not finding.resolved
                for finding in red_team.findings
            )
            and sensitivity.validation_id == validation.validation_id
            and robustness.validation_id == validation.validation_id
            and robustness.sensitivity_id == sensitivity.sensitivity_id
            and red_team.validation_id == validation.validation_id
            and red_team.sensitivity_id == sensitivity.sensitivity_id
            and red_team.robustness_id == robustness.robustness_id
        )

    def _records(
        self,
        state: ProblemState,
        context: ResultContext,
        validation: ValidationReport,
        sensitivity: SensitivityReport,
        robustness: RobustnessReport,
        red_team: RedTeamReport,
    ) -> list[EvidenceRecord]:
        model = context.model
        result = context.result
        common = {
            "model_id": model.model_id,
            "model_version": model.version,
            "model_digest": result.model_digest,
            "result_id": result.result_id,
            "solver_run_id": result.solver_run_id,
            "execution_record_id": result.execution_record_id,
            "validation_id": validation.validation_id,
            "sensitivity_id": sensitivity.sensitivity_id,
            "robustness_id": robustness.robustness_id,
            "red_team_report_id": red_team.report_id,
            "is_mock": False,
        }
        records: list[EvidenceRecord] = []
        for item in state.evidence_items:
            if item.status is not EvidenceStatus.ACCEPTED:
                continue
            evidence_type = {
                "FACT": EvidenceType.PROBLEM_FACT,
                "DATA": EvidenceType.DATA_FACT,
            }.get(item.type.value)
            if evidence_type is None:
                continue
            records.append(
                self._record(
                    state,
                    evidence_type,
                    "problem_analysis",
                    item.evidence_id,
                    str(state.version),
                    item.content,
                    item.model_dump(mode="json"),
                    [f"problem_state:{state.version}", item.evidence_id],
                    common,
                )
            )
        records.append(
            self._record(
                state,
                EvidenceType.MODEL,
                "mathematical_model",
                str(model.model_id),
                str(model.version),
                model.description,
                {
                    "name": model.name,
                    "model_family": model.model_family.value,
                    "limitations": model.limitations,
                    "confidence": model.confidence,
                },
                [f"model:{model.model_id}:v{model.version}"],
                common,
            )
        )
        for assumption in model.assumptions:
            if not assumption.supported:
                continue
            records.append(
                self._record(
                    state,
                    EvidenceType.ASSUMPTION,
                    "mathematical_model",
                    assumption.assumption_id,
                    str(model.version),
                    assumption.statement,
                    assumption.model_dump(mode="json"),
                    assumption.source_refs,
                    common,
                )
            )
        for equation in model.equations:
            records.append(
                self._record(
                    state,
                    EvidenceType.EQUATION,
                    "mathematical_model",
                    equation.equation_id,
                    str(model.version),
                    equation.meaning,
                    equation.model_dump(mode="json"),
                    equation.source_refs,
                    common,
                )
            )
        objective_unit = (
            model.objective.unit.display if model.objective and model.objective.unit else None
        )
        records.extend(
            [
                self._record(
                    state,
                    EvidenceType.RESULT,
                    "solver_result",
                    str(result.result_id),
                    "1",
                    "verified formal solver result",
                    {
                        "objective": {"value": result.objective, "unit": objective_unit},
                        "key_outputs": result.key_outputs,
                        "solver": result.solver.value,
                        "solver_version": context.solver_run.solver_version,
                        "status": result.status.value,
                        "is_optimal": result.status is SolverStatus.OPTIMAL,
                        "is_feasible": context.solver_run.result.is_feasible,
                        "model_family": model.model_family.value,
                    },
                    result.evidence_refs,
                    common,
                ),
                self._report_record(
                    state, EvidenceType.VALIDATION, validation, validation.validation_id, common
                ),
                self._report_record(
                    state,
                    EvidenceType.SENSITIVITY,
                    sensitivity,
                    sensitivity.sensitivity_id,
                    common,
                ),
                self._report_record(
                    state,
                    EvidenceType.ROBUSTNESS,
                    robustness,
                    robustness.robustness_id,
                    common,
                ),
                self._report_record(
                    state,
                    EvidenceType.RED_TEAM,
                    red_team,
                    red_team.report_id,
                    common,
                ),
            ]
        )
        return records

    def _report_record(
        self,
        state: ProblemState,
        evidence_type: EvidenceType,
        report: ValidationReport | SensitivityReport | RobustnessReport | RedTeamReport,
        report_id: UUID,
        common: dict[str, Any],
    ) -> EvidenceRecord:
        payload = report.model_dump(mode="json")
        return self._record(
            state,
            evidence_type,
            report.__class__.__name__,
            str(report_id),
            "1",
            f"verified {evidence_type.value.casefold()} report",
            payload,
            [f"{evidence_type.value.casefold()}:{report_id}"],
            common,
        )

    @staticmethod
    def _record(
        state: ProblemState,
        evidence_type: EvidenceType,
        source_type: str,
        source_id: str,
        source_version: str,
        summary: str,
        payload: dict[str, Any],
        source_refs: list[str],
        common: dict[str, Any],
    ) -> EvidenceRecord:
        source_hash = sha256_json(payload)
        evidence_id = uuid5(
            NAMESPACE_URL,
            f"mathmodel-ai:{state.project_id}:{source_type}:{source_id}:{source_version}:{source_hash}",
        )
        return EvidenceRecord(
            evidence_id=evidence_id,
            project_id=state.project_id,
            problem_id=state.problem_id,
            evidence_type=evidence_type,
            source_type=source_type,
            source_id=source_id,
            source_version=source_version,
            content_summary=summary,
            structured_payload=payload,
            verified=True,
            verification_status=EvidenceVerificationStatus.VERIFIED,
            provenance=EvidenceProvenance(
                source_refs=source_refs, source_hash=source_hash, **common
            ),
        )

    @staticmethod
    def _snapshot(
        model: MathematicalModel,
        result_id: UUID,
        validation: ValidationReport,
        sensitivity: SensitivityReport,
        robustness: RobustnessReport,
        red_team: RedTeamReport,
        records: tuple[EvidenceRecord, ...],
    ) -> EvidenceSnapshot:
        evidence_ids = [item.evidence_id for item in records]
        digest = next(
            item.provenance.model_digest for item in records if item.provenance.model_digest
        )
        snapshot = EvidenceSnapshot(
            verified_model_id=model.model_id,
            verified_model_version=model.version,
            verified_model_digest=digest,
            verified_result_id=result_id,
            validation_id=validation.validation_id,
            sensitivity_id=sensitivity.sensitivity_id,
            robustness_id=robustness.robustness_id,
            red_team_report_id=red_team.report_id,
            evidence_ids=evidence_ids,
            snapshot_hash="0" * 64,
        )
        return snapshot.model_copy(
            update={"snapshot_hash": calculate_snapshot_hash(snapshot, records)}
        )
