from __future__ import annotations

import re
import unicodedata
from io import BytesIO
from typing import Any, ClassVar

import pymupdf
from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from mathmodel_ai.core.errors import StorageError
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.registry import EquationRegistry
from mathmodel_ai.paper.bundle import PaperBundleBuilder
from mathmodel_ai.paper.compiler import PaperCompilation
from mathmodel_ai.paper.evidence import calculate_snapshot_hash
from mathmodel_ai.paper.hashing import sha256_bytes, sha256_json, sha256_text
from mathmodel_ai.paper.literature import claim_digest, reference_digest
from mathmodel_ai.paper.registry import CitationRegistry, FigureRegistry, TableRegistry
from mathmodel_ai.paper.rendering import BibTeXRenderer, RenderedPaper, escape_latex
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.paper import (
    CitationMetadataCheck,
    CitationSupportCheck,
    Claim,
    ClaimImportance,
    EvidenceRecord,
    EvidenceSnapshot,
    EvidenceType,
    FigureRecord,
    FigureType,
    PaperArtifact,
    PaperArtifactKind,
    PaperIR,
    PaperManifest,
    PaperSectionType,
    PaperValidationIssue,
    PaperValidationSeverity,
    PaperVersion,
    ProblemRequirement,
    ReferenceRecord,
    TableRecord,
)


def _issue(code: str, message: str, object_ref: str | None = None) -> PaperValidationIssue:
    return PaperValidationIssue(
        code=code,
        message=message,
        severity=PaperValidationSeverity.ERROR,
        object_ref=object_ref,
    )


class EvidenceSnapshotValidator:
    """Rebuild snapshot truth from leaf records and the exact model revision."""

    def validate(
        self,
        snapshot: EvidenceSnapshot,
        records: list[EvidenceRecord],
        model: MathematicalModel,
        equations: EquationRegistry,
    ) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        record_tuple = tuple(records)
        if snapshot.evidence_ids != [item.evidence_id for item in records]:
            issues.append(_issue("EVIDENCE_SNAPSHOT_STALE", "snapshot evidence order/set changed"))
        if len(snapshot.evidence_ids) != len(set(snapshot.evidence_ids)):
            issues.append(_issue("EVIDENCE_SNAPSHOT_STALE", "snapshot contains duplicate evidence"))
        if calculate_snapshot_hash(snapshot, record_tuple) != snapshot.snapshot_hash:
            issues.append(_issue("EVIDENCE_SNAPSHOT_STALE", "snapshot digest does not recompute"))
        if (
            snapshot.verified_model_id != model.model_id
            or snapshot.verified_model_version != model.version
            or snapshot.verified_model_digest != mathematical_model_digest(model)
            or equations.model_id != model.model_id
            or equations.version != model.version
        ):
            issues.append(
                _issue(
                    "WRONG_EQUATION_VERSION",
                    "paper model/equation registry does not match the verified snapshot revision",
                )
            )
        result_records = [item for item in records if item.evidence_type is EvidenceType.RESULT]
        if len(result_records) != 1 or result_records[0].source_id != str(
            snapshot.verified_result_id
        ):
            issues.append(
                _issue(
                    "WRONG_VERIFIED_RESULT",
                    "snapshot does not contain exactly its selected formal result",
                )
            )
        for record in records:
            if (
                not record.verified
                or record.provenance.is_mock
                or record.provenance.source_hash != sha256_json(record.structured_payload)
            ):
                issues.append(
                    _issue(
                        "UNVERIFIED_EVIDENCE",
                        "evidence leaf is unverified, Mock, revoked, or content-tampered",
                        str(record.evidence_id),
                    )
                )
            provenance = record.provenance
            if (
                provenance.model_id != snapshot.verified_model_id
                or provenance.model_version != snapshot.verified_model_version
                or provenance.model_digest != snapshot.verified_model_digest
                or provenance.result_id != snapshot.verified_result_id
            ):
                issues.append(
                    _issue(
                        "EVIDENCE_SNAPSHOT_STALE",
                        "evidence provenance differs from snapshot identity",
                        str(record.evidence_id),
                    )
                )
            if record.evidence_type is EvidenceType.EQUATION and record.source_version != str(
                snapshot.verified_model_version
            ):
                issues.append(
                    _issue(
                        "WRONG_EQUATION_VERSION",
                        "equation evidence came from another model version",
                        record.source_id,
                    )
                )
        return issues


class RenderedContentValidator:
    """Independently compare PaperIR/registries with emitted TeX and BibTeX."""

    _CITE = re.compile(r"\\cite\{([^}]+)\}")

    def validate(
        self,
        paper: PaperIR,
        rendered: RenderedPaper,
        equations: EquationRegistry,
        figures: FigureRegistry,
        tables: TableRegistry,
        citations: CitationRegistry,
    ) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        if (
            sha256_text(rendered.tex) != rendered.tex_hash
            or sha256_text(rendered.bibliography) != rendered.bib_hash
        ):
            issues.append(_issue("DOCUMENT_INTEGRITY_ERROR", "rendered source hash mismatch"))
        for section in [*paper.sections, *paper.appendices]:
            marker = rf"\section{{{escape_latex(section.title)}}}\label{{{section.section_id}}}"
            if marker not in rendered.tex:
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "PaperIR section is absent from rendered LaTeX",
                        section.section_id,
                    )
                )
        for claim in paper.claims:
            referenced = any(
                block.claim_ref == claim.claim_id
                for block in [
                    *paper.abstract,
                    *(
                        block
                        for section in [*paper.sections, *paper.appendices]
                        for block in section.blocks
                    ),
                ]
            )
            if referenced and escape_latex(claim.text) not in rendered.tex:
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "claim text is absent or drifted in rendered LaTeX",
                        claim.claim_id,
                    )
                )
        for section in [*paper.sections, *paper.appendices]:
            for equation_id in section.equation_refs:
                equation = equations.lookup(equation_id)
                expected = (
                    f"{equation.latex}\\label{{{equation_id}}}" if equation is not None else ""
                )
                if not expected or expected not in rendered.tex:
                    issues.append(
                        _issue(
                            "DOCUMENT_INTEGRITY_ERROR",
                            "registered equation is absent or semantically drifted",
                            equation_id,
                        )
                    )
            for figure_id in section.figure_refs:
                if (
                    rf"\includegraphics[width=0.85\linewidth]{{{figure_id}.png}}"
                    not in rendered.tex
                ):
                    issues.append(
                        _issue(
                            "DOCUMENT_INTEGRITY_ERROR",
                            "required figure is not rendered",
                            figure_id,
                        )
                    )
            for table_id in section.table_refs:
                if rf"\label{{{table_id}}}" not in rendered.tex:
                    issues.append(
                        _issue(
                            "DOCUMENT_INTEGRITY_ERROR", "required table is not rendered", table_id
                        )
                    )
        expected_bib = BibTeXRenderer().render(citations)
        if rendered.bibliography != expected_bib:
            issues.append(
                _issue("DOCUMENT_INTEGRITY_ERROR", "BibTeX differs from verified references")
            )
        actual_citations = {
            key for match in self._CITE.finditer(rendered.tex) for key in match.group(1).split(",")
        }
        expected_citations = {
            reference for claim in paper.claims for reference in claim.citation_refs
        } | {
            reference
            for section in [*paper.sections, *paper.appendices]
            for reference in section.citation_refs
        }
        if actual_citations != expected_citations:
            issues.append(
                _issue("DOCUMENT_INTEGRITY_ERROR", "rendered citation keys differ from PaperIR")
            )
        return issues


class CitationFreshnessValidator:
    def validate(
        self,
        claims: list[Claim],
        references: list[ReferenceRecord],
        metadata_checks: list[CitationMetadataCheck],
        support_checks: list[CitationSupportCheck],
    ) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        claim_map = {item.claim_id: item for item in claims}
        reference_map = {item.reference_id: item for item in references}
        for reference in references:
            if reference.is_mock:
                issues.append(
                    _issue(
                        "MOCK_EVIDENCE",
                        "Mock literature cannot support a final paper",
                        reference.reference_id,
                    )
                )
        for metadata_check in metadata_checks:
            metadata_reference = reference_map.get(metadata_check.reference_id)
            if metadata_reference is None or metadata_check.reference_digest != reference_digest(
                metadata_reference
            ):
                issues.append(
                    _issue(
                        "REFERENCE_METADATA_CONFLICT",
                        "metadata verification is stale for the current reference",
                        metadata_check.reference_id,
                    )
                )
        for support_check in support_checks:
            claim = claim_map.get(support_check.claim_id)
            support_reference = reference_map.get(support_check.reference_id)
            if (
                claim is None
                or support_reference is None
                or support_check.claim_digest != claim_digest(claim)
                or support_check.reference_digest != reference_digest(support_reference)
            ):
                issues.append(
                    _issue(
                        "REFERENCE_NOT_SUPPORTED",
                        "citation support review is stale for claim/reference content",
                        support_check.claim_id,
                    )
                )
        return issues


class AssetSemanticIntegrityValidator:
    _FIGURE_TYPE: ClassVar[dict[FigureType, EvidenceType]] = {
        FigureType.SENSITIVITY_CURVE: EvidenceType.SENSITIVITY,
        FigureType.ROBUSTNESS_DISTRIBUTION: EvidenceType.ROBUSTNESS,
        FigureType.SCENARIO_COMPARISON: EvidenceType.ROBUSTNESS,
    }

    def validate(
        self,
        figures: list[FigureRecord],
        tables: list[TableRecord],
        evidence: list[EvidenceRecord],
    ) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        by_id = {item.evidence_id: item for item in evidence}
        assets: list[FigureRecord | TableRecord] = [*figures, *tables]
        for asset in assets:
            sources = [by_id.get(reference) for reference in asset.source_evidence_refs]
            if any(item is None for item in sources):
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "asset references unknown evidence",
                        self._id(asset),
                    )
                )
                continue
            concrete = [item for item in sources if item is not None]
            expected_types = [item.evidence_type for item in concrete]
            expected_binding = [
                {
                    "evidence_id": str(item.evidence_id),
                    "evidence_type": item.evidence_type.value,
                    "source_id": item.source_id,
                    "source_hash": item.provenance.source_hash,
                }
                for item in concrete
            ]
            if (
                asset.source_evidence_types != expected_types
                or asset.source_binding.get("evidence") != expected_binding
            ):
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "asset semantic evidence binding does not recompute",
                        self._id(asset),
                    )
                )
        for figure in figures:
            required = self._FIGURE_TYPE.get(figure.figure_type)
            if required is not None and required not in figure.source_evidence_types:
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        f"{figure.figure_type.value} requires {required.value} evidence",
                        figure.figure_id,
                    )
                )
            if required is EvidenceType.SENSITIVITY:
                source = by_id[figure.source_evidence_refs[0]]
                if figure.data_payload.get("reviewed_response") is True:
                    payload = source.structured_payload
                    values = payload.get("reviewed_metric_values", {})
                    replay_ids = payload.get("reviewed_replay_ids", {})
                    metric_ids = figure.data_payload.get("metric_ids", [])
                    scenario_ids = figure.data_payload.get("scenario_ids", [])
                    x = figure.data_payload.get("x", [])
                    y = figure.data_payload.get("y", [])
                    execution_ids = figure.source_binding.get("experiment_ids", [])
                    valid = (
                        isinstance(values, dict)
                        and isinstance(replay_ids, dict)
                        and len(metric_ids)
                        == len(scenario_ids)
                        == len(x)
                        == len(y)
                        == len(execution_ids)
                        and len(metric_ids) >= 2
                        and len(set(metric_ids)) == len(metric_ids)
                        and len(set(scenario_ids)) == len(scenario_ids)
                    )
                    if valid:
                        try:
                            valid = all(
                                x[index] == index + 1
                                and float(y[index]) == float(values[metric_id])
                                and str(execution_ids[index])
                                == str(replay_ids[scenario_ids[index]])
                                for index, metric_id in enumerate(metric_ids)
                            )
                        except (KeyError, TypeError, ValueError):
                            valid = False
                    if not valid:
                        issues.append(
                            _issue(
                                "DOCUMENT_INTEGRITY_ERROR",
                                "reviewed sensitivity figure differs from verified replay values",
                                figure.figure_id,
                            )
                        )
                    continue
                experiments = [
                    item
                    for item in source.structured_payload.get("experiments", [])
                    if isinstance(item, dict)
                ]
                known_experiments = {str(item.get("experiment_id")) for item in experiments}
                bound = set(map(str, figure.source_binding.get("experiment_ids", [])))
                if not bound or not bound <= known_experiments:
                    issues.append(
                        _issue(
                            "DOCUMENT_INTEGRITY_ERROR",
                            "sensitivity figure experiment binding is missing or stale",
                            figure.figure_id,
                        )
                    )
                known_parameters = {
                    str(perturbation.get("symbol"))
                    for experiment in experiments
                    for perturbation in experiment.get("perturbations", [])
                    if isinstance(perturbation, dict)
                }
                raw_parameters = figure.source_binding.get("parameter", [])
                bound_parameters = {
                    str(item)
                    for item in (
                        raw_parameters if isinstance(raw_parameters, list) else [raw_parameters]
                    )
                    if item is not None
                }
                expected_points = {
                    (
                        float(perturbation["fraction"]),
                        float(experiment["objective_value"]),
                    )
                    for experiment in experiments
                    if experiment.get("objective_value") is not None
                    for perturbation in experiment.get("perturbations", [])[:1]
                    if isinstance(perturbation, dict)
                }
                try:
                    actual_points = set(
                        zip(
                            map(float, figure.data_payload.get("x", [])),
                            map(float, figure.data_payload.get("y", [])),
                            strict=True,
                        )
                    )
                except (TypeError, ValueError):
                    actual_points = set()
                if (
                    not bound_parameters
                    or not bound_parameters <= known_parameters
                    or actual_points != expected_points
                ):
                    issues.append(
                        _issue(
                            "DOCUMENT_INTEGRITY_ERROR",
                            "sensitivity figure parameter/metric data differs from experiments",
                            figure.figure_id,
                        )
                    )
        for table in tables:
            title = table.title.casefold()
            if (
                "sensitivity" in title
                and EvidenceType.SENSITIVITY not in table.source_evidence_types
            ):
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "sensitivity table uses wrong evidence type",
                        table.table_id,
                    )
                )
            if "robust" in title and EvidenceType.ROBUSTNESS not in table.source_evidence_types:
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "robustness table uses wrong evidence type",
                        table.table_id,
                    )
                )
            if "symbol" in title and EvidenceType.MODEL not in table.source_evidence_types:
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "symbol table uses wrong evidence type",
                        table.table_id,
                    )
                )
            if EvidenceType.RESULT in table.source_evidence_types:
                result = by_id[table.source_evidence_refs[0]].structured_payload.get("objective")
                objective = result.get("value") if isinstance(result, dict) else result
                rows = {str(row[0]).casefold(): row[1] for row in table.rows if len(row) >= 2}
                if "objective" in rows and rows["objective"] != objective:
                    issues.append(
                        _issue(
                            "CROSS_ARTIFACT_VALUE_CONFLICT",
                            "result table objective differs from evidence",
                            table.table_id,
                        )
                    )
        return issues

    @staticmethod
    def _id(asset: FigureRecord | TableRecord) -> str:
        return asset.figure_id if isinstance(asset, FigureRecord) else asset.table_id


class SubproblemCoverageValidator:
    def validate(
        self, paper: PaperIR, requirements: list[ProblemRequirement]
    ) -> list[PaperValidationIssue]:
        expected = {item.subproblem_id: item for item in requirements}
        actual = {item.subproblem_id: item for item in paper.subproblem_coverage}
        issues: list[PaperValidationIssue] = []
        if expected.keys() != actual.keys():
            for missing in sorted(expected.keys() - actual.keys()):
                issues.append(
                    _issue("MISSING_SUBPROBLEM", "required subproblem is absent", missing)
                )
            for extra in sorted(actual.keys() - expected.keys()):
                issues.append(
                    _issue("DOCUMENT_INTEGRITY_ERROR", "unknown subproblem coverage", extra)
                )
        sections = {item.section_id: item for item in [*paper.sections, *paper.appendices]}
        claims = {item.claim_id for item in paper.claims}
        for subproblem_id, record in actual.items():
            requirement = expected.get(subproblem_id)
            if requirement is None:
                continue
            if set(record.required_outputs) != set(requirement.required_outputs):
                issues.append(
                    _issue(
                        "MISSING_SUBPROBLEM",
                        "required outputs are not fully covered",
                        subproblem_id,
                    )
                )
            if any(
                section_id not in sections
                or subproblem_id not in sections[section_id].subproblem_refs
                for section_id in record.section_ids
            ) or any(claim_id not in claims for claim_id in record.claim_refs):
                issues.append(
                    _issue(
                        "MISSING_SUBPROBLEM",
                        "coverage points to missing paper objects",
                        subproblem_id,
                    )
                )
        return issues


class DocumentCompletenessValidator:
    _ABSTRACT_ROLES: ClassVar[set[str]] = {
        "PROBLEM",
        "METHOD",
        "KEY_RESULT",
        "CONCLUSION",
    }

    def validate(self, paper: PaperIR) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        present_sections = {item.section_type for item in paper.sections}
        if paper.abstract:
            present_sections.add(PaperSectionType.ABSTRACT)
        if paper.bibliography:
            present_sections.add(PaperSectionType.REFERENCES)
        for required in paper.competition_profile.required_sections:
            if required not in present_sections:
                issues.append(
                    _issue("MISSING_REQUIRED_SECTION", "required section is absent", required.value)
                )
        if any(not section.blocks for section in paper.sections):
            issues.append(_issue("EMPTY_PAPER", "paper contains an empty formal section"))
        referenced_claims = {
            block.claim_ref
            for block in [
                *paper.abstract,
                *(block for section in paper.sections for block in section.blocks),
            ]
            if block.claim_ref is not None
        }
        for claim in paper.claims:
            if (
                claim.importance in {ClaimImportance.CRITICAL, ClaimImportance.MAJOR}
                and claim.claim_id not in referenced_claims
            ):
                issues.append(
                    _issue(
                        "REQUIRED_CLAIM_MISSING", "material claim is not rendered", claim.claim_id
                    )
                )
        roles = {
            block.abstract_role.value for block in paper.abstract if block.abstract_role is not None
        }
        if roles != self._ABSTRACT_ROLES:
            issues.append(
                _issue(
                    "ABSTRACT_COVERAGE_MISSING",
                    "abstract must cover problem, method, key result, and conclusion",
                )
            )
        key_result_claims = {
            block.claim_ref
            for block in paper.abstract
            if block.abstract_role is not None and block.abstract_role.value == "KEY_RESULT"
        }
        if not any(
            claim.claim_id in key_result_claims and claim.importance is ClaimImportance.CRITICAL
            for claim in paper.claims
        ):
            issues.append(
                _issue(
                    "ABSTRACT_COVERAGE_MISSING",
                    "abstract key result is not a critical evidence-backed claim",
                )
            )
        return issues


class ArtifactIntegrityValidator:
    """Re-read final bytes and bind all paper artifacts to one version and manifest."""

    def __init__(self, store: FileStore) -> None:
        self._store = store

    def validate(
        self,
        *,
        version: PaperVersion,
        rendered: RenderedPaper,
        compilation: PaperCompilation,
        manifest: PaperManifest | None,
        artifacts: list[PaperArtifact],
        references: list[ReferenceRecord],
        figures: list[FigureRecord],
        tables: list[TableRecord],
    ) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        by_kind: dict[PaperArtifactKind, list[PaperArtifact]] = {}
        artifact_bytes: dict[Any, bytes] = {}
        for artifact in artifacts:
            by_kind.setdefault(artifact.kind, []).append(artifact)
            if (
                artifact.project_id != version.project_id
                or artifact.problem_id != version.problem_id
                or artifact.paper_id != version.paper_id
                or artifact.paper_version != version.version
            ):
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "artifact identity/version mismatch",
                        str(artifact.artifact_id),
                    )
                )
            try:
                data = self._store.read_bytes(artifact.storage_key)
            except (FileNotFoundError, OSError, StorageError, ValueError):
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "artifact is missing or unreadable",
                        str(artifact.artifact_id),
                    )
                )
                continue
            artifact_bytes[artifact.artifact_id] = data
            if len(data) != artifact.size_bytes or sha256_bytes(data) != artifact.sha256:
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "artifact bytes differ from immutable record",
                        str(artifact.artifact_id),
                    )
                )
        singleton = [
            PaperArtifactKind.TEX,
            PaperArtifactKind.BIB,
            PaperArtifactKind.PDF,
            PaperArtifactKind.MANIFEST,
        ]
        if any(len(by_kind.get(kind, [])) != 1 for kind in singleton):
            issues.append(
                _issue(
                    "DOCUMENT_INTEGRITY_ERROR",
                    "final bundle requires one TeX/Bib/PDF/manifest artifact",
                )
            )
            return issues
        tex = by_kind[PaperArtifactKind.TEX][0]
        bib = by_kind[PaperArtifactKind.BIB][0]
        pdf = by_kind[PaperArtifactKind.PDF][0]
        manifest_artifact = by_kind[PaperArtifactKind.MANIFEST][0]
        if artifact_bytes.get(tex.artifact_id) != rendered.tex.encode(
            "utf-8"
        ) or artifact_bytes.get(bib.artifact_id) != rendered.bibliography.encode("utf-8"):
            issues.append(
                _issue("DOCUMENT_INTEGRITY_ERROR", "stored source differs from rendered source")
            )
        record = compilation.record
        if (
            (record.paper_id, record.paper_version) != (version.paper_id, version.version)
            or record.tex_hash != tex.sha256
            or record.bib_hash != bib.sha256
            or record.pdf_artifact_id != pdf.artifact_id
            or record.page_count is None
            or record.fatal_warnings
        ):
            issues.append(
                _issue("DOCUMENT_INTEGRITY_ERROR", "compile record does not bind final artifacts")
            )
        if manifest is None:
            issues.append(_issue("DOCUMENT_INTEGRITY_ERROR", "final manifest is missing"))
            return issues
        raw_manifest = artifact_bytes.get(manifest_artifact.artifact_id)
        try:
            persisted_manifest = PaperManifest.model_validate_json(raw_manifest or b"")
        except (ValidationError, ValueError):
            issues.append(
                _issue("DOCUMENT_INTEGRITY_ERROR", "manifest artifact is invalid or tampered")
            )
            return issues
        if persisted_manifest != manifest or not PaperBundleBuilder.verify_manifest(manifest):
            issues.append(_issue("DOCUMENT_INTEGRITY_ERROR", "manifest digest/content mismatch"))
        expected = {
            "paper_ir_hash": sha256_json(
                version.paper_ir.model_dump(mode="json", exclude={"status"})
            ),
            "claim_set_hash": sha256_json(
                [item.model_dump(mode="json") for item in version.paper_ir.claims]
            ),
            "document_registry_hash": sha256_json(
                [item.model_dump(mode="json") for item in version.paper_ir.document_registry]
            ),
            "reference_set_hash": sha256_json(
                [item.model_dump(mode="json") for item in references]
            ),
            "figure_set_hash": sha256_json([item.model_dump(mode="json") for item in figures]),
            "table_set_hash": sha256_json([item.model_dump(mode="json") for item in tables]),
            "tex_artifact_id": tex.artifact_id,
            "bib_artifact_id": bib.artifact_id,
            "pdf_artifact_id": pdf.artifact_id,
            "tex_hash": tex.sha256,
            "bib_hash": bib.sha256,
            "pdf_hash": pdf.sha256,
        }
        if any(getattr(manifest, key) != value for key, value in expected.items()):
            issues.append(
                _issue(
                    "DOCUMENT_INTEGRITY_ERROR", "manifest does not bind current records/artifacts"
                )
            )
        pdf_bytes = artifact_bytes.get(pdf.artifact_id)
        if pdf_bytes is not None:
            issues.extend(self._pdf_content(version.paper_ir, pdf_bytes, record.page_count))
        return issues

    @staticmethod
    def _pdf_content(
        paper: PaperIR, pdf: bytes, expected_pages: int | None
    ) -> list[PaperValidationIssue]:
        try:
            reader = PdfReader(BytesIO(pdf), strict=True)
            page_count = len(reader.pages)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        except (PdfReadError, ValueError, TypeError, KeyError):
            return [_issue("DOCUMENT_INTEGRITY_ERROR", "final PDF failed strict parsing")]
        issues: list[PaperValidationIssue] = []
        if not page_count or page_count != expected_pages:
            issues.append(_issue("DOCUMENT_INTEGRITY_ERROR", "final PDF page count changed"))
        normalized_pdf = _normalize_pdf_claim_text(text)
        material_claims = [
            claim
            for claim in paper.claims
            if claim.importance in {ClaimImportance.CRITICAL, ClaimImportance.MAJOR}
        ]
        missing = [
            claim
            for claim in material_claims
            if _normalize_pdf_claim_text(claim.text) not in normalized_pdf
        ]
        if missing:
            # pypdf can collapse spaces around TeX-escaped identifiers; use an
            # independent extractor on the same strictly parsed PDF bytes.
            try:
                with pymupdf.open(stream=pdf, filetype="pdf") as document:  # type: ignore[no-untyped-call]
                    alternate = (
                        _normalize_pdf_claim_text(" ".join(page.get_text() for page in document))
                        if len(document) == page_count
                        else ""
                    )
            except (RuntimeError, ValueError, TypeError):
                alternate = ""
            for claim in missing:
                if _normalize_pdf_claim_text(claim.text) not in alternate:
                    issues.append(
                        _issue(
                            "DOCUMENT_INTEGRITY_ERROR",
                            "material claim is absent from final PDF",
                            claim.claim_id,
                        )
                    )
        return issues


def _normalize_pdf_claim_text(value: str) -> str:
    """Ignore TeX hyphenation/possessive extraction noise, not words or numbers."""
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    # A PDF line break after a date hyphen or numeric minus is layout noise;
    # retain the sign/hyphen and every following digit exactly.
    value = re.sub(r"-\s*\n\s*(?=\d)", "-", value)
    value = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", value)
    value = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", value)
    # Some TeX font encodings extract the apostrophe in an English possessive
    # as two replacement glyphs; this does not alter words or numeric tokens.
    value = re.sub(r"(?<=[A-Za-z])(?:'|\u2019|\ufffd{1,2})(?=s\b)", "", value)
    return " ".join(value.split())
