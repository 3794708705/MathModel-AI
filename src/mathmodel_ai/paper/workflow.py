from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel

from mathmodel_ai.agents import (
    AgentRunResult,
    AgentRunStatus,
    CitationAgent,
    LiteratureAgent,
    PaperAgent,
    PaperFactualAuditAgent,
)
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.registry import EquationRegistry, SymbolRegistry
from mathmodel_ai.paper.assets import FigureAgent, SymbolTableBuilder, TableAgent
from mathmodel_ai.paper.bundle import PaperBundleBuilder
from mathmodel_ai.paper.claims import (
    ClaimEvidenceGraph,
    ClaimEvidenceValidator,
    resolve_source_field,
)
from mathmodel_ai.paper.compiler import PaperCompilation, PDFCompiler
from mathmodel_ai.paper.evidence import VerifiedEvidenceBuilder, VerifiedEvidenceBundle
from mathmodel_ai.paper.integrity import (
    ArtifactIntegrityValidator,
    AssetSemanticIntegrityValidator,
    EvidenceSnapshotValidator,
    RenderedContentValidator,
    SubproblemCoverageValidator,
)
from mathmodel_ai.paper.literature import (
    CitationMetadataVerifier,
    CitationSupportVerifier,
    LiteratureSource,
)
from mathmodel_ai.paper.registry import (
    CitationRegistry,
    DocumentRegistry,
    FigureRegistry,
    TableRegistry,
)
from mathmodel_ai.paper.rendering import LaTeXRenderer, RenderedPaper
from mathmodel_ai.paper.repository import PaperRepository
from mathmodel_ai.paper.validators import PaperQualityGate
from mathmodel_ai.reasoning.state_machine import ensure_transition
from mathmodel_ai.routing.schemas import EscalationLevel, TaskProfile, TaskType
from mathmodel_ai.schemas.paper import (
    CitationAgentInput,
    CitationMetadataCheck,
    CitationSupportCheck,
    Claim,
    ClaimEvidenceLink,
    ClaimEvidenceSupport,
    ComparisonClaimValue,
    CompetitionProfile,
    DocumentObjectType,
    EvidenceRecord,
    EvidenceType,
    FigureRecord,
    FigureType,
    LiteratureAgentInput,
    LiteraturePlan,
    NumericClaimValue,
    PaperAgentInput,
    PaperArtifact,
    PaperCompileStatus,
    PaperFactualAuditDraft,
    PaperFactualAuditInput,
    PaperIR,
    PaperManifest,
    PaperQualityReport,
    PaperQualityStatus,
    PaperValidationIssue,
    PaperValidationSeverity,
    PaperVersion,
    PaperVersionRef,
    ProblemRequirement,
    ReferenceRecord,
    TableRecord,
)
from mathmodel_ai.schemas.problem_state import (
    ProblemState,
    WorkflowStage,
    WorkflowStatus,
)


class PaperWorkflowError(ValueError):
    pass


@dataclass(frozen=True)
class PaperWorkflowOutcome:
    state: ProblemState
    version: PaperVersion
    quality: PaperQualityReport
    compilation: PaperCompilation
    references: tuple[ReferenceRecord, ...]
    figures: tuple[FigureRecord, ...]
    tables: tuple[TableRecord, ...]
    rendered: RenderedPaper


class PaperWorkflow:
    def __init__(
        self,
        *,
        repository: PaperRepository,
        evidence_builder: VerifiedEvidenceBuilder,
        literature_agent: LiteratureAgent,
        literature_source: LiteratureSource,
        citation_agent: CitationAgent,
        paper_agent: PaperAgent,
        audit_agent: PaperFactualAuditAgent,
        figure_agent: FigureAgent,
        table_agent: TableAgent,
        renderer: LaTeXRenderer,
        compiler: PDFCompiler,
        bundle_builder: PaperBundleBuilder,
        store: FileStore,
    ) -> None:
        self._repository = repository
        self._evidence_builder = evidence_builder
        self._literature_agent = literature_agent
        self._literature_source = literature_source
        self._citation_agent = citation_agent
        self._paper_agent = paper_agent
        self._audit_agent = audit_agent
        self._figure_agent = figure_agent
        self._table_agent = table_agent
        self._renderer = renderer
        self._compiler = compiler
        self._bundle_builder = bundle_builder
        self._store = store

    async def run(
        self,
        project_id: UUID,
        *,
        competition_profile: CompetitionProfile | None = None,
    ) -> PaperWorkflowOutcome:
        bundle = self._evidence_builder.build(project_id)
        ensure_transition(bundle.state.current_stage, WorkflowStage.PAPER)
        paper_id, paper_version = self._repository.next_identity(project_id)
        references, metadata_checks, literature_run = await self._literature(bundle)
        literature_plan = self._require_output(literature_run, "LiteratureAgent")
        self._repository.persist_literature(
            project_id=project_id,
            problem_id=bundle.state.problem_id,
            source=self._literature_source.name,
            status=(
                "VERIFIED"
                if references
                and all(item.metadata_status.value == "VERIFIED" for item in references)
                else "PARTIAL"
            ),
            plan=literature_plan,
            references=references,
        )
        bundle = self._evidence_builder.attach_references(bundle, references)
        figures, tables, asset_artifacts = self._assets(bundle, paper_id, paper_version)
        paper, paper_run = await self._author(
            bundle,
            paper_id,
            paper_version,
            references,
            figures,
            tables,
            competition_profile or CompetitionProfile(),
        )
        registry = self._document_registry(bundle, paper, references, figures, tables)
        paper = paper.model_copy(update={"document_registry": registry.entries()})
        graph = self._claim_graph(paper.claims, list(bundle.records))
        claims = ClaimEvidenceValidator().apply_statuses(graph)
        paper = paper.model_copy(update={"claims": claims})
        registry = self._document_registry(bundle, paper, references, figures, tables)
        paper = paper.model_copy(update={"document_registry": registry.entries()})
        graph = self._claim_graph(claims, list(bundle.records))
        support_checks = await self._citation_support(
            bundle.state,
            claims,
            references,
        )
        audit_run = await self._audit(bundle.state, paper, claims, list(bundle.records))
        rendered = self._renderer.render(
            paper=paper,
            claims=claims,
            equations=EquationRegistry.from_model(bundle.context.model),
            figures=FigureRegistry(figures),
            tables=TableRegistry(tables),
            citations=CitationRegistry(references),
        )
        figure_images = self._figure_images(figures, asset_artifacts)
        compilation = self._compiler.compile(
            rendered,
            project_id=project_id,
            problem_id=bundle.state.problem_id,
            paper_id=paper_id,
            paper_version=paper_version,
            figure_images=figure_images,
        )
        all_artifacts = [*asset_artifacts, *compilation.artifacts]
        independent_passed, independent_mock = self._audit_status(audit_run, claims)
        provisional = PaperVersion(
            paper_id=paper_id,
            project_id=project_id,
            problem_id=bundle.state.problem_id,
            version=paper_version,
            parent_version=paper_version - 1 if paper_version > 1 else None,
            revision_reason="evidence-grounded Phase 6 paper build",
            evidence_snapshot=bundle.snapshot,
            paper_ir=paper.model_copy(update={"status": PaperQualityStatus.VALIDATING}),
            status=PaperQualityStatus.VALIDATING,
            paper_agent_run_id=paper_run.run_id,
            paper_agent_is_mock=paper_run.is_mock,
        )
        manifest: PaperManifest | None = None
        manifest_hash: str | None = None
        if compilation.record.status is PaperCompileStatus.SUCCEEDED:
            manifest, manifest_artifact = self._bundle_builder.build_manifest(
                version=provisional,
                references=references,
                figures=figures,
                tables=tables,
                artifacts=all_artifacts,
            )
            if not self._bundle_builder.verify_manifest(manifest):
                raise PaperWorkflowError("deterministic paper manifest verification failed")
            manifest_hash = manifest.manifest_hash
            all_artifacts.append(manifest_artifact)
        equations = EquationRegistry.from_model(bundle.context.model)
        requirements = [
            ProblemRequirement(
                subproblem_id=item.subproblem_id,
                required_outputs=item.output_required,
            )
            for item in bundle.state.subproblems
        ]
        integrity_issues = self._artifact_integrity(figures, tables, all_artifacts)
        integrity_issues.extend(
            EvidenceSnapshotValidator().validate(
                bundle.snapshot,
                list(bundle.records),
                bundle.context.model,
                equations,
            )
        )
        integrity_issues.extend(
            RenderedContentValidator().validate(
                paper,
                rendered,
                equations,
                FigureRegistry(figures),
                TableRegistry(tables),
                CitationRegistry(references),
            )
        )
        integrity_issues.extend(
            AssetSemanticIntegrityValidator().validate(figures, tables, list(bundle.records))
        )
        integrity_issues.extend(SubproblemCoverageValidator().validate(paper, requirements))
        integrity_issues.extend(
            ArtifactIntegrityValidator(self._store).validate(
                version=provisional,
                rendered=rendered,
                compilation=compilation,
                manifest=manifest,
                artifacts=all_artifacts,
                references=references,
                figures=figures,
                tables=tables,
            )
        )
        expected_symbol_units = {
            item.symbol: item.unit.display if item.unit else None
            for item in SymbolRegistry.from_model(bundle.context.model).definitions
        }
        quality = PaperQualityGate().evaluate(
            paper=paper,
            graph=graph,
            model=bundle.context.model,
            equations=equations,
            figures=FigureRegistry(figures),
            tables=TableRegistry(tables),
            citations=CitationRegistry(references),
            metadata_checks=metadata_checks,
            support_checks=support_checks,
            compile_succeeded=compilation.record.status is PaperCompileStatus.SUCCEEDED,
            artifact_integrity_issues=integrity_issues,
            declared_symbols={
                str(row[0]): (
                    str(row[1]),
                    (
                        None
                        if str(row[2]) == "1" and expected_symbol_units.get(str(row[0])) is None
                        else str(row[2])
                    ),
                )
                for table in tables
                if "symbol" in table.title.casefold()
                for row in table.rows
                if len(row) >= 3
            },
            independent_review_passed=independent_passed,
            independent_review_is_mock=independent_mock,
        )
        paper = paper.model_copy(update={"status": quality.status})
        version = PaperVersion(
            paper_id=paper_id,
            project_id=project_id,
            problem_id=bundle.state.problem_id,
            version=paper_version,
            parent_version=paper_version - 1 if paper_version > 1 else None,
            revision_reason="evidence-grounded Phase 6 paper build",
            evidence_snapshot=bundle.snapshot,
            paper_ir=paper,
            status=quality.status,
            manifest_hash=manifest_hash,
            paper_agent_run_id=paper_run.run_id,
            paper_agent_is_mock=paper_run.is_mock,
        )
        state = self._state(bundle.state, version, quality, compilation)
        self._repository.persist(
            version=version,
            state=state,
            quality=quality,
            compile_record=compilation.record,
            evidence=list(bundle.records),
            claims=claims,
            links=graph.links,
            references=references,
            support_checks=support_checks,
            figures=figures,
            tables=tables,
            registry=paper.document_registry,
            artifacts=all_artifacts,
        )
        return PaperWorkflowOutcome(
            state=state,
            version=version,
            quality=quality,
            compilation=compilation,
            references=tuple(references),
            figures=tuple(figures),
            tables=tuple(tables),
            rendered=rendered,
        )

    async def search_literature(
        self, project_id: UUID
    ) -> tuple[list[ReferenceRecord], list[CitationMetadataCheck], LiteraturePlan]:
        bundle = self._evidence_builder.build(project_id)
        references, checks, run = await self._literature(bundle)
        plan = self._require_output(run, "LiteratureAgent")
        self._repository.persist_literature(
            project_id=project_id,
            problem_id=bundle.state.problem_id,
            source=self._literature_source.name,
            status=(
                "VERIFIED"
                if references
                and all(item.metadata_status.value == "VERIFIED" for item in references)
                else "PARTIAL"
            ),
            plan=plan,
            references=references,
        )
        return references, checks, plan

    async def _literature(
        self, bundle: VerifiedEvidenceBundle
    ) -> tuple[list[ReferenceRecord], list[CitationMetadataCheck], AgentRunResult[LiteraturePlan]]:
        run = await self._literature_agent.run(
            LiteratureAgentInput(
                problem_title=bundle.state.title,
                problem_summary=bundle.state.raw_problem,
                model_name=bundle.context.model.name,
                model_family=bundle.context.model.model_family.value,
                evidence_summaries=[item.content_summary for item in bundle.records],
                requested_uses=[],
            ),
            bundle.state,
            TaskProfile(
                task_type=TaskType.PAPER_IR,
                complexity=3,
                reasoning_requirement=3,
                long_context_requirement=3,
                review_requirement=2,
                minimum_level=EscalationLevel.FLAGSHIP_HIGH,
            ),
        )
        plan = self._require_output(run, "LiteratureAgent")
        found: dict[str, ReferenceRecord] = {}
        for need in plan.needs:
            for reference in await self._literature_source.search(need, bundle.state.project_id):
                if reference.project_id != bundle.state.project_id:
                    raise PaperWorkflowError("literature source returned another project identity")
                found.setdefault(reference.reference_id, reference)
        verifier = CitationMetadataVerifier()
        references: list[ReferenceRecord] = []
        checks: list[CitationMetadataCheck] = []
        for candidate in found.values():
            check = await verifier.verify(candidate, self._literature_source)
            checks.append(check)
            references.append(verifier.verified_copy(candidate, check))
        if not references:
            raise PaperWorkflowError("paper workflow requires retrieved literature metadata")
        CitationRegistry(references)
        return references, checks, run

    def _assets(
        self,
        bundle: VerifiedEvidenceBundle,
        paper_id: UUID,
        paper_version: int,
    ) -> tuple[list[FigureRecord], list[TableRecord], list[PaperArtifact]]:
        evidence = list(bundle.records)
        sensitivity_evidence = next(
            item for item in evidence if item.evidence_type is EvidenceType.SENSITIVITY
        )
        result_evidence = next(
            item for item in evidence if item.evidence_type is EvidenceType.RESULT
        )
        model_evidence = next(item for item in evidence if item.evidence_type is EvidenceType.MODEL)
        points = [
            (experiment.perturbations[0].fraction, experiment.objective_value)
            for experiment in bundle.sensitivity.experiments
            if experiment.perturbations and experiment.objective_value is not None
        ]
        if len(points) < 2:
            raise PaperWorkflowError("verified sensitivity report cannot produce a figure")
        figure, figure_artifacts = self._figure_agent.generate(
            project_id=bundle.state.project_id,
            problem_id=bundle.state.problem_id,
            paper_id=paper_id,
            paper_version=paper_version,
            figure_id="FIG-001",
            title="Sensitivity response",
            caption="Verified objective values under parameter perturbations.",
            figure_type=FigureType.SENSITIVITY_CURVE,
            data_payload={
                "x": [item[0] for item in points],
                "y": [item[1] for item in points],
                "x_label": "perturbation fraction",
                "y_label": "objective",
                "metric": "objective",
                "parameter": sorted(
                    {
                        perturbation.symbol
                        for experiment in bundle.sensitivity.experiments
                        for perturbation in experiment.perturbations
                    }
                ),
                "experiment_ids": [
                    str(experiment.experiment_id) for experiment in bundle.sensitivity.experiments
                ],
            },
            source_evidence_refs=[sensitivity_evidence.evidence_id],
            evidence=evidence,
        )
        result_table, result_artifact = self._table_agent.generate(
            project_id=bundle.state.project_id,
            problem_id=bundle.state.problem_id,
            paper_id=paper_id,
            paper_version=paper_version,
            table_id="TAB-001",
            title="Verified solver result",
            caption="Formal result selected by verified_result_id.",
            columns=["Metric", "Value"],
            rows=[
                ["Solver", bundle.context.result.solver.value],
                ["Status", bundle.context.result.status.value],
                ["Objective", bundle.context.result.objective],
            ],
            source_evidence_refs=[result_evidence.evidence_id],
            evidence=evidence,
        )
        columns, rows = SymbolTableBuilder.rows(bundle.context.model)
        symbol_table, symbol_artifact = self._table_agent.generate(
            project_id=bundle.state.project_id,
            problem_id=bundle.state.problem_id,
            paper_id=paper_id,
            paper_version=paper_version,
            table_id="TAB-002",
            title="Symbol registry",
            caption="Symbols generated from the verified mathematical model registry.",
            columns=columns,
            rows=rows,
            source_evidence_refs=[model_evidence.evidence_id],
            evidence=evidence,
        )
        return (
            [figure],
            [result_table, symbol_table],
            [*figure_artifacts, result_artifact, symbol_artifact],
        )

    async def _author(
        self,
        bundle: VerifiedEvidenceBundle,
        paper_id: UUID,
        paper_version: int,
        references: list[ReferenceRecord],
        figures: list[FigureRecord],
        tables: list[TableRecord],
        profile: CompetitionProfile,
    ) -> tuple[PaperIR, AgentRunResult[PaperIR]]:
        run = await self._paper_agent.run(
            PaperAgentInput(
                project_id=bundle.state.project_id,
                assigned_paper_id=paper_id,
                assigned_version=paper_version,
                title=bundle.state.title,
                evidence=list(bundle.records),
                evidence_snapshot=bundle.snapshot,
                equation_ids=[item.equation_id for item in bundle.context.model.equations],
                figure_ids=[item.figure_id for item in figures],
                table_ids=[item.table_id for item in tables],
                reference_ids=[item.reference_id for item in references],
                required_subproblems=[
                    ProblemRequirement(
                        subproblem_id=item.subproblem_id,
                        required_outputs=item.output_required,
                    )
                    for item in bundle.state.subproblems
                ],
                competition_profile=profile,
            ),
            bundle.state,
            TaskProfile(
                task_type=TaskType.PAPER_IR,
                complexity=4,
                reasoning_requirement=4,
                math_requirement=3,
                long_context_requirement=5,
                review_requirement=4,
                blast_radius=4,
                minimum_level=EscalationLevel.FLAGSHIP_HIGH,
            ),
        )
        paper = self._require_output(run, "PaperAgent")
        if (paper.paper_id, paper.version) != (paper_id, paper_version):
            raise PaperWorkflowError("PaperAgent changed assigned paper identity/version")
        if paper.evidence_snapshot != bundle.snapshot:
            raise PaperWorkflowError("PaperAgent changed the verified evidence snapshot")
        if not set(paper.bibliography) <= {item.reference_id for item in references}:
            raise PaperWorkflowError("PaperAgent referenced unverified bibliography metadata")
        return paper.model_copy(update={"status": PaperQualityStatus.DRAFT}), run

    @staticmethod
    def _document_registry(
        bundle: VerifiedEvidenceBundle,
        paper: PaperIR,
        references: list[ReferenceRecord],
        figures: list[FigureRecord],
        tables: list[TableRecord],
    ) -> DocumentRegistry:
        registry = DocumentRegistry.from_model(bundle.context.model)
        for reference in references:
            registry.add(
                DocumentObjectType.REFERENCE,
                source_id=reference.reference_id,
                object_id=reference.reference_id,
                content=reference.model_dump(mode="json"),
            )
        for figure in figures:
            registry.add(
                DocumentObjectType.FIGURE,
                source_id=figure.figure_id,
                object_id=figure.figure_id,
                content=figure.model_dump(mode="json"),
            )
        for table in tables:
            registry.add(
                DocumentObjectType.TABLE,
                source_id=table.table_id,
                object_id=table.table_id,
                content=table.model_dump(mode="json"),
            )
        for section in [*paper.sections, *paper.appendices]:
            registry.add(
                DocumentObjectType.SECTION,
                source_id=section.section_id,
                object_id=section.section_id,
                content=section.model_dump(mode="json"),
            )
        for claim in paper.claims:
            registry.add(
                DocumentObjectType.CLAIM,
                source_id=claim.claim_id,
                object_id=claim.claim_id,
                content=claim.model_dump(mode="json"),
            )
        return registry

    @staticmethod
    def _claim_graph(claims: list[Claim], evidence: list[EvidenceRecord]) -> ClaimEvidenceGraph:
        graph = ClaimEvidenceGraph(evidence)
        by_id = {item.evidence_id: item for item in evidence}
        for claim in claims:
            graph.add_claim(claim)
            value = claim.structured_value
            fields: list[str | None]
            if isinstance(value, NumericClaimValue):
                fields = [value.source_field]
            elif isinstance(value, ComparisonClaimValue):
                fields = [value.baseline_source_field, value.verified_source_field]
            else:
                fields = [None]
            linked_any = False
            for reference in claim.evidence_refs:
                evidence_item = by_id[reference]
                for source_field in fields:
                    if source_field is not None:
                        try:
                            resolve_source_field(evidence_item.structured_payload, source_field)
                        except (KeyError, ValueError):
                            continue
                    graph.add_link(
                        ClaimEvidenceLink(
                            project_id=claim.project_id,
                            claim_id=claim.claim_id,
                            evidence_id=reference,
                            support=ClaimEvidenceSupport.DIRECT_SUPPORT,
                            source_field=source_field,
                            rationale="explicit PaperAgent claim-to-evidence binding",
                        )
                    )
                    linked_any = True
            if claim.evidence_refs and not linked_any:
                raise PaperWorkflowError(
                    f"claim {claim.claim_id} source fields do not resolve in linked evidence"
                )
        return graph

    async def _citation_support(
        self,
        state: ProblemState,
        claims: list[Claim],
        references: list[ReferenceRecord],
    ) -> list[CitationSupportCheck]:
        by_id = {item.reference_id: item for item in references}
        checks: list[CitationSupportCheck] = []
        verifier = CitationSupportVerifier()
        for claim in claims:
            for reference_id in claim.citation_refs:
                reference = by_id.get(reference_id)
                if reference is None:
                    continue
                trusted_text = reference.trusted_excerpt or reference.abstract
                if not trusted_text:
                    continue
                run = await self._citation_agent.run(
                    CitationAgentInput(
                        claim=claim,
                        reference=reference,
                        trusted_text=trusted_text,
                    ),
                    state,
                    TaskProfile(
                        task_type=TaskType.CITATION_VERIFICATION,
                        complexity=4,
                        reasoning_requirement=4,
                        long_context_requirement=3,
                        review_requirement=5,
                        blast_radius=4,
                        minimum_level=EscalationLevel.FLAGSHIP_XHIGH,
                    ),
                )
                draft = self._require_output(run, "CitationAgent")
                checks.append(
                    verifier.verify(
                        claim=claim,
                        reference=reference,
                        draft=draft,
                        reviewer_is_mock=run.is_mock,
                    )
                )
        return checks

    async def _audit(
        self,
        state: ProblemState,
        paper: PaperIR,
        claims: list[Claim],
        evidence: list[EvidenceRecord],
    ) -> AgentRunResult[PaperFactualAuditDraft]:
        return await self._audit_agent.run(
            PaperFactualAuditInput(paper=paper, claims=claims, evidence=evidence),
            state,
            TaskProfile(
                task_type=TaskType.FINAL_ACCEPTANCE,
                complexity=5,
                reasoning_requirement=5,
                math_requirement=4,
                long_context_requirement=5,
                review_requirement=5,
                blast_radius=5,
                minimum_level=EscalationLevel.FLAGSHIP_MAX,
            ),
        )

    @staticmethod
    def _audit_status(
        run: AgentRunResult[PaperFactualAuditDraft], claims: list[Claim]
    ) -> tuple[bool, bool]:
        if run.status is not AgentRunStatus.SUCCEEDED or run.output is None:
            return False, run.is_mock
        material = {
            item.claim_id for item in claims if item.importance.value in {"CRITICAL", "MAJOR"}
        }
        reviewed = set(run.output.reviewed_claim_ids)
        return run.output.passed and material <= reviewed and not run.output.findings, run.is_mock

    def _figure_images(
        self, figures: list[FigureRecord], artifacts: list[PaperArtifact]
    ) -> dict[str, bytes]:
        by_id = {item.artifact_id: item for item in artifacts}
        return {
            f"{figure.figure_id}.png": self._store.read_bytes(
                by_id[figure.image_artifact_id].storage_key
            )
            for figure in figures
        }

    def _artifact_integrity(
        self,
        figures: list[FigureRecord],
        tables: list[TableRecord],
        artifacts: list[PaperArtifact],
    ) -> list[PaperValidationIssue]:
        by_id = {item.artifact_id: item for item in artifacts}
        issues: list[PaperValidationIssue] = []
        figure_registry = FigureRegistry(figures)
        for figure in figures:
            errors = figure_registry.verify(
                figure.figure_id,
                data_bytes=self._store.read_bytes(by_id[figure.data_artifact_id].storage_key),
                code_bytes=self._store.read_bytes(by_id[figure.code_artifact_id].storage_key),
                image_bytes=self._store.read_bytes(by_id[figure.image_artifact_id].storage_key),
            )
            issues.extend(
                PaperValidationIssue(
                    code=error.split(":", 1)[0],
                    message=error,
                    severity=PaperValidationSeverity.ERROR,
                    object_ref=figure.figure_id,
                )
                for error in errors
            )
        table_registry = TableRegistry(tables)
        for table in tables:
            errors = table_registry.verify(
                table.table_id,
                data_bytes=self._store.read_bytes(by_id[table.data_artifact_id].storage_key),
            )
            issues.extend(
                PaperValidationIssue(
                    code=error.split(":", 1)[0],
                    message=error,
                    severity=PaperValidationSeverity.ERROR,
                    object_ref=table.table_id,
                )
                for error in errors
            )
        return issues

    @staticmethod
    def _state(
        current: ProblemState,
        version: PaperVersion,
        quality: PaperQualityReport,
        compilation: PaperCompilation,
    ) -> ProblemState:
        status = (
            WorkflowStatus.SUCCEEDED
            if quality.status is PaperQualityStatus.READY_FOR_FINAL_JURY
            else WorkflowStatus.HUMAN_REVIEW
            if quality.status is PaperQualityStatus.HUMAN_REVIEW
            else WorkflowStatus.FAILED
        )
        return current.model_copy(
            update={
                "schema_version": 6,
                "version": current.version + 1,
                "paper_versions": [
                    *current.paper_versions,
                    PaperVersionRef(
                        paper_id=version.paper_id,
                        version=version.version,
                        verified_result_id=version.evidence_snapshot.verified_result_id,
                        evidence_snapshot_hash=version.evidence_snapshot.snapshot_hash,
                        status=version.status,
                        manifest_hash=version.manifest_hash,
                    ),
                ],
                "paper_state": {
                    "paper_id": str(version.paper_id),
                    "version": version.version,
                    "status": version.status.value,
                    "compile_status": compilation.record.status.value,
                    "manifest_hash": version.manifest_hash,
                },
                "current_stage": WorkflowStage.PAPER,
                "status": status,
                "updated_by": "paper_workflow",
                "update_reason": "Phase 6 evidence-grounded paper pipeline completed",
                "updated_at": datetime.now(UTC),
            }
        )

    @staticmethod
    def _require_output[OutputT: BaseModel](
        run: AgentRunResult[OutputT], agent_name: str
    ) -> OutputT:
        if run.status is not AgentRunStatus.SUCCEEDED or run.output is None:
            raise PaperWorkflowError(f"{agent_name} did not produce a usable structured output")
        return run.output
