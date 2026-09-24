from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.core.errors import ResourceNotFoundError
from mathmodel_ai.db.models import (
    CitationSupportRecord,
    ClaimEvidenceLinkRecord,
    ClaimRecord,
    DocumentRegistryRecord,
    EvidenceRecordModel,
    FigureRecordModel,
    LiteratureSearchRecord,
    PaperArtifactRecord,
    PaperSectionRecord,
    PaperVersionRecord,
    Problem,
    ProblemStateRecord,
    ReferenceRecordModel,
    TableRecordModel,
)
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.schemas.paper import (
    CitationSupportCheck,
    Claim,
    ClaimEvidenceLink,
    DocumentRegistryEntry,
    EvidenceRecord,
    FigureRecord,
    LiteraturePlan,
    PaperArtifact,
    PaperCompileRecord,
    PaperQualityReport,
    PaperVersion,
    ReferenceMetadataStatus,
    ReferenceRecord,
    TableRecord,
)
from mathmodel_ai.schemas.problem_state import ProblemState


class PaperRepository:
    """Atomic Phase 6 persistence for one immutable paper/evidence snapshot."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def next_identity(self, project_id: UUID) -> tuple[UUID, int]:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(PaperVersionRecord)
                .where(PaperVersionRecord.project_id == project_id)
                .order_by(PaperVersionRecord.created_at.desc(), PaperVersionRecord.version.desc())
                .limit(1)
            )
            if row is None:
                problem_id = session.scalar(
                    select(Problem.id).where(Problem.project_id == project_id)
                )
                if problem_id is None:
                    raise ResourceNotFoundError(f"project {project_id} was not found")
                return uuid4(), 1
            return row.paper_id, row.version + 1

    def canonicalize_evidence(
        self, project_id: UUID, evidence: tuple[EvidenceRecord, ...]
    ) -> tuple[EvidenceRecord, ...]:
        """Reuse immutable evidence records when only construction time differs."""
        identifiers = [item.evidence_id for item in evidence]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate paper evidence identifier")
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(EvidenceRecordModel).where(EvidenceRecordModel.id.in_(identifiers))
            )
            existing = {row.id: row for row in rows}
        canonical: list[EvidenceRecord] = []
        for item in evidence:
            row = existing.get(item.evidence_id)
            if row is None:
                canonical.append(item)
                continue
            if row.project_id != project_id:
                raise ValueError("immutable evidence belongs to a different project")
            stored = EvidenceRecord.model_validate(row.record_json)
            if stored.model_dump(mode="json", exclude={"created_at"}) != item.model_dump(
                mode="json", exclude={"created_at"}
            ):
                raise ValueError("immutable evidence identifier has conflicting content")
            canonical.append(stored)
        return tuple(canonical)

    def persist(
        self,
        *,
        version: PaperVersion,
        state: ProblemState,
        quality: PaperQualityReport,
        compile_record: PaperCompileRecord,
        evidence: list[EvidenceRecord],
        claims: list[Claim],
        links: list[ClaimEvidenceLink],
        references: list[ReferenceRecord],
        support_checks: list[CitationSupportCheck],
        figures: list[FigureRecord],
        tables: list[TableRecord],
        registry: list[DocumentRegistryEntry],
        artifacts: list[PaperArtifact],
    ) -> None:
        if version.status is not quality.status or version.paper_ir.status is not quality.status:
            raise ValueError("paper version, IR, and quality status must match")
        if compile_record.paper_id != version.paper_id or (
            compile_record.paper_version != version.version
        ):
            raise ValueError("compile record belongs to a different paper version")
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, version.project_id)
            self._validate_next_revision(current, state)
            paper_row = PaperVersionRecord(
                id=uuid4(),
                paper_id=version.paper_id,
                project_id=version.project_id,
                problem_id=version.problem_id,
                version=version.version,
                parent_version=version.parent_version,
                verified_result_id=version.evidence_snapshot.verified_result_id,
                evidence_snapshot_id=version.evidence_snapshot.snapshot_id,
                evidence_snapshot_hash=version.evidence_snapshot.snapshot_hash,
                status=version.status.value,
                manifest_hash=version.manifest_hash,
                paper_agent_run_id=version.paper_agent_run_id,
                paper_agent_is_mock=version.paper_agent_is_mock,
                version_json=version.model_dump(mode="json"),
                quality_json=quality.model_dump(mode="json"),
                compile_json=compile_record.model_dump(mode="json"),
                created_at=version.created_at,
            )
            session.add(paper_row)
            session.flush()
            for evidence_item in evidence:
                existing_evidence = session.get(EvidenceRecordModel, evidence_item.evidence_id)
                if existing_evidence is not None:
                    if existing_evidence.record_json != evidence_item.model_dump(mode="json"):
                        raise ValueError("immutable evidence identifier has conflicting content")
                    continue
                session.add(
                    EvidenceRecordModel(
                        id=evidence_item.evidence_id,
                        project_id=evidence_item.project_id,
                        problem_id=evidence_item.problem_id,
                        paper_version_record_id=paper_row.id,
                        evidence_type=evidence_item.evidence_type.value,
                        source_type=evidence_item.source_type,
                        source_id=evidence_item.source_id,
                        source_version=evidence_item.source_version,
                        source_hash=evidence_item.provenance.source_hash,
                        verified=evidence_item.verified,
                        verification_status=evidence_item.verification_status.value,
                        record_json=evidence_item.model_dump(mode="json"),
                        created_at=evidence_item.created_at,
                    )
                )
            claim_rows: dict[str, UUID] = {}
            for claim in claims:
                row_id = uuid4()
                claim_rows[claim.claim_id] = row_id
                session.add(
                    ClaimRecord(
                        id=row_id,
                        project_id=claim.project_id,
                        paper_version_record_id=paper_row.id,
                        claim_id=claim.claim_id,
                        claim_type=claim.claim_type.value,
                        importance=claim.importance.value,
                        verification_status=claim.verification_status.value,
                        claim_json=claim.model_dump(mode="json"),
                    )
                )
            session.flush()
            for link in links:
                claim_row_id = claim_rows.get(link.claim_id)
                if claim_row_id is None:
                    raise ValueError("claim/evidence link references an unpersisted claim")
                session.add(
                    ClaimEvidenceLinkRecord(
                        id=link.link_id,
                        project_id=link.project_id,
                        claim_record_id=claim_row_id,
                        evidence_record_id=link.evidence_id,
                        support=link.support.value,
                        source_field=link.source_field,
                        link_json=link.model_dump(mode="json"),
                    )
                )
            for reference in references:
                existing_reference = session.scalar(
                    select(ReferenceRecordModel).where(
                        ReferenceRecordModel.project_id == reference.project_id,
                        ReferenceRecordModel.reference_id == reference.reference_id,
                    )
                )
                if existing_reference is not None:
                    if existing_reference.reference_json != reference.model_dump(mode="json"):
                        raise ValueError("immutable reference identifier has conflicting content")
                    continue
                session.add(
                    ReferenceRecordModel(
                        project_id=reference.project_id,
                        reference_id=reference.reference_id,
                        doi=reference.doi,
                        source=reference.source.value,
                        source_id=reference.source_id,
                        metadata_status=reference.metadata_status.value,
                        reference_json=reference.model_dump(mode="json"),
                    )
                )
            for support_check in support_checks:
                session.add(
                    CitationSupportRecord(
                        id=support_check.check_id,
                        project_id=support_check.project_id,
                        paper_version_record_id=paper_row.id,
                        claim_id=support_check.claim_id,
                        reference_id=support_check.reference_id,
                        status=support_check.status.value,
                        reviewer_is_mock=support_check.reviewer_is_mock,
                        check_json=support_check.model_dump(mode="json"),
                    )
                )
            for section in [*version.paper_ir.sections, *version.paper_ir.appendices]:
                session.add(
                    PaperSectionRecord(
                        paper_version_record_id=paper_row.id,
                        section_id=section.section_id,
                        section_type=section.section_type.value,
                        order=section.order,
                        section_json=section.model_dump(mode="json"),
                    )
                )
            for figure in figures:
                session.add(
                    FigureRecordModel(
                        project_id=figure.project_id,
                        paper_version_record_id=paper_row.id,
                        figure_id=figure.figure_id,
                        data_hash=figure.data_hash,
                        code_hash=figure.code_hash,
                        image_hash=figure.image_hash,
                        figure_json=figure.model_dump(mode="json"),
                    )
                )
            for table in tables:
                session.add(
                    TableRecordModel(
                        project_id=table.project_id,
                        paper_version_record_id=paper_row.id,
                        table_id=table.table_id,
                        data_hash=table.data_hash,
                        table_json=table.model_dump(mode="json"),
                    )
                )
            for entry in registry:
                session.add(
                    DocumentRegistryRecord(
                        paper_version_record_id=paper_row.id,
                        object_id=entry.object_id,
                        object_type=entry.object_type.value,
                        content_hash=entry.content_hash,
                        entry_json=entry.model_dump(mode="json"),
                    )
                )
            for artifact in artifacts:
                session.add(
                    PaperArtifactRecord(
                        id=artifact.artifact_id,
                        project_id=artifact.project_id,
                        paper_version_record_id=paper_row.id,
                        kind=artifact.kind.value,
                        name=artifact.name,
                        mime_type=artifact.mime_type,
                        size_bytes=artifact.size_bytes,
                        sha256=artifact.sha256,
                        storage_key=artifact.storage_key,
                        created_at=artifact.created_at,
                    )
                )
            self._replace_state(session, current, state)

    def get_version(
        self, project_id: UUID, paper_id: UUID | None = None, version: int | None = None
    ) -> PaperVersion:
        with session_scope(self._session_factory) as session:
            statement = select(PaperVersionRecord).where(
                PaperVersionRecord.project_id == project_id
            )
            if paper_id is not None:
                statement = statement.where(PaperVersionRecord.paper_id == paper_id)
            if version is not None:
                statement = statement.where(PaperVersionRecord.version == version)
            else:
                statement = statement.order_by(
                    PaperVersionRecord.created_at.desc(), PaperVersionRecord.version.desc()
                ).limit(1)
            row = session.scalar(statement)
            if row is None:
                raise ResourceNotFoundError("paper version was not found")
            return PaperVersion.model_validate(row.version_json)

    def persist_literature(
        self,
        *,
        project_id: UUID,
        problem_id: UUID,
        source: str,
        status: str,
        plan: LiteraturePlan,
        references: list[ReferenceRecord],
    ) -> None:
        with session_scope(self._session_factory) as session:
            session.add(
                LiteratureSearchRecord(
                    project_id=project_id,
                    problem_id=problem_id,
                    source=source,
                    status=status,
                    plan_json=plan.model_dump(mode="json"),
                )
            )
            for reference in references:
                existing = session.scalar(
                    select(ReferenceRecordModel).where(
                        ReferenceRecordModel.project_id == project_id,
                        ReferenceRecordModel.reference_id == reference.reference_id,
                    )
                )
                if existing is None:
                    session.add(
                        ReferenceRecordModel(
                            project_id=project_id,
                            reference_id=reference.reference_id,
                            doi=reference.doi,
                            source=reference.source.value,
                            source_id=reference.source_id,
                            metadata_status=reference.metadata_status.value,
                            reference_json=reference.model_dump(mode="json"),
                        )
                    )
                elif existing.reference_json != reference.model_dump(mode="json"):
                    raise ValueError("immutable reference identifier has conflicting content")

    def list_references(self, project_id: UUID) -> list[ReferenceRecord]:
        with session_scope(self._session_factory) as session:
            rows = list(
                session.scalars(
                    select(ReferenceRecordModel)
                    .where(ReferenceRecordModel.project_id == project_id)
                    .order_by(ReferenceRecordModel.reference_id)
                )
            )
            return [ReferenceRecord.model_validate(row.reference_json) for row in rows]

    def latest_literature(
        self, project_id: UUID, *, source: str
    ) -> tuple[LiteraturePlan, list[ReferenceRecord]] | None:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(LiteratureSearchRecord)
                .where(
                    LiteratureSearchRecord.project_id == project_id,
                    LiteratureSearchRecord.source == source,
                    LiteratureSearchRecord.status == "VERIFIED",
                )
                .order_by(LiteratureSearchRecord.created_at.desc())
                .limit(1)
            )
            if row is None:
                return None
            plan = LiteraturePlan.model_validate(row.plan_json)
        references = self.list_references(project_id)
        if not references or any(
            item.project_id != project_id
            or item.metadata_status is not ReferenceMetadataStatus.VERIFIED
            for item in references
        ):
            return None
        return plan, references

    def canonicalize_references(
        self, project_id: UUID, references: list[ReferenceRecord]
    ) -> list[ReferenceRecord]:
        """Reuse verified immutable metadata across paper-only attempts."""
        previous = {item.reference_id: item for item in self.list_references(project_id)}
        canonical = []
        for reference in references:
            existing = previous.get(reference.reference_id)
            if existing is None:
                canonical.append(reference)
                continue
            if (
                existing.project_id != project_id
                or existing.model_dump(mode="json", exclude={"retrieved_at"})
                != reference.model_dump(mode="json", exclude={"retrieved_at"})
            ):
                raise ValueError("retrieved literature metadata conflicts with immutable reference")
            canonical.append(existing)
        return canonical

    def get_quality(
        self,
        project_id: UUID,
        paper_id: UUID | None = None,
        version: int | None = None,
    ) -> PaperQualityReport:
        row = self._get_paper_row(project_id, paper_id, version)
        if row.quality_json is None:
            raise ResourceNotFoundError("paper quality report was not found")
        return PaperQualityReport.model_validate(row.quality_json)

    def get_compile(
        self,
        project_id: UUID,
        paper_id: UUID | None = None,
        version: int | None = None,
    ) -> PaperCompileRecord:
        row = self._get_paper_row(project_id, paper_id, version)
        if row.compile_json is None:
            raise ResourceNotFoundError("paper compile record was not found")
        return PaperCompileRecord.model_validate(row.compile_json)

    def _get_paper_row(
        self,
        project_id: UUID,
        paper_id: UUID | None,
        version: int | None = None,
    ) -> PaperVersionRecord:
        with session_scope(self._session_factory) as session:
            statement = select(PaperVersionRecord).where(
                PaperVersionRecord.project_id == project_id
            )
            if paper_id is not None:
                statement = statement.where(PaperVersionRecord.paper_id == paper_id)
            if version is not None:
                statement = statement.where(PaperVersionRecord.version == version)
            row = session.scalar(
                statement.order_by(
                    PaperVersionRecord.created_at.desc(), PaperVersionRecord.version.desc()
                ).limit(1)
            )
            if row is None:
                raise ResourceNotFoundError("paper version was not found")
            session.expunge(row)
            return row

    def list_version_artifacts(
        self,
        project_id: UUID,
        paper_id: UUID,
        version: int,
    ) -> list[PaperArtifact]:
        """Load artifacts for one explicit paper version; never substitute latest."""

        with session_scope(self._session_factory) as session:
            paper_row = session.scalar(
                select(PaperVersionRecord).where(
                    PaperVersionRecord.project_id == project_id,
                    PaperVersionRecord.paper_id == paper_id,
                    PaperVersionRecord.version == version,
                )
            )
            if paper_row is None:
                raise ResourceNotFoundError("paper version was not found")
            rows = list(
                session.scalars(
                    select(PaperArtifactRecord)
                    .where(PaperArtifactRecord.paper_version_record_id == paper_row.id)
                    .order_by(PaperArtifactRecord.created_at, PaperArtifactRecord.id)
                )
            )
            return [
                PaperArtifact(
                    artifact_id=row.id,
                    project_id=row.project_id,
                    problem_id=paper_row.problem_id,
                    paper_id=paper_row.paper_id,
                    paper_version=paper_row.version,
                    kind=row.kind,
                    name=row.name,
                    mime_type=row.mime_type,
                    size_bytes=row.size_bytes,
                    sha256=row.sha256,
                    storage_key=row.storage_key,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def list_artifacts(self, project_id: UUID, paper_id: UUID) -> list[PaperArtifact]:
        with session_scope(self._session_factory) as session:
            paper_rows = select(PaperVersionRecord.id).where(
                PaperVersionRecord.project_id == project_id,
                PaperVersionRecord.paper_id == paper_id,
            )
            rows = list(
                session.scalars(
                    select(PaperArtifactRecord)
                    .where(PaperArtifactRecord.paper_version_record_id.in_(paper_rows))
                    .order_by(PaperArtifactRecord.created_at)
                )
            )
            return [
                PaperArtifact(
                    artifact_id=row.id,
                    project_id=row.project_id,
                    problem_id=self._problem_id_for_paper(session, row.paper_version_record_id),
                    paper_id=paper_id,
                    paper_version=self._version_for_paper(session, row.paper_version_record_id),
                    kind=row.kind,
                    name=row.name,
                    mime_type=row.mime_type,
                    size_bytes=row.size_bytes,
                    sha256=row.sha256,
                    storage_key=row.storage_key,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    @staticmethod
    def _problem_id_for_paper(session: Session, record_id: UUID) -> UUID:
        row = session.get(PaperVersionRecord, record_id)
        if row is None:
            raise ResourceNotFoundError("paper version was not found")
        return row.problem_id

    @staticmethod
    def _version_for_paper(session: Session, record_id: UUID) -> int:
        row = session.get(PaperVersionRecord, record_id)
        if row is None:
            raise ResourceNotFoundError("paper version was not found")
        return row.version

    @staticmethod
    def _lock_current(session: Session, project_id: UUID) -> ProblemStateRecord:
        row = session.scalar(
            select(ProblemStateRecord)
            .join(Problem, Problem.id == ProblemStateRecord.problem_id)
            .where(Problem.project_id == project_id, ProblemStateRecord.is_current.is_(True))
            .with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError(f"project {project_id} was not found")
        return row

    @staticmethod
    def _validate_next_revision(current: ProblemStateRecord, state: ProblemState) -> None:
        if current.revision + 1 != state.version:
            raise ValueError(
                f"stale state update: current={current.revision}, requested={state.version}"
            )

    @staticmethod
    def _replace_state(session: Session, current: ProblemStateRecord, state: ProblemState) -> None:
        current.is_current = False
        session.flush()
        session.add(
            ProblemStateRecord(
                problem_id=state.problem_id,
                revision=state.version,
                schema_version=state.schema_version,
                current_stage=state.current_stage.value,
                status=state.status.value,
                is_current=True,
                state_json=state.model_dump(mode="json"),
                updated_by=state.updated_by,
                update_reason=state.update_reason,
            )
        )
