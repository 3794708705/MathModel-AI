from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from mathmodel_ai.api.schemas import FinalRunRequest, FinalRunResponse
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    FinalJuryReport,
    SubmissionArtifact,
    SubmissionCheckResult,
    SubmissionSnapshot,
    SubmissionStatus,
)
from mathmodel_ai.submission.package import SubmissionIntegrityVerifier, SubmissionPackage
from mathmodel_ai.submission.profiles import CompetitionProfileRegistry
from mathmodel_ai.submission.repository import SubmissionRepository
from mathmodel_ai.submission.workflow import FinalSubmissionWorkflow

router = APIRouter(prefix="/api/v1", tags=["final-submission"])


def _services(
    request: Request,
) -> tuple[SubmissionRepository, CompetitionProfileRegistry, FinalSubmissionWorkflow]:
    repository: SubmissionRepository = request.app.state.submission_repository
    profiles: CompetitionProfileRegistry = request.app.state.competition_profile_registry
    workflow: FinalSubmissionWorkflow = request.app.state.final_submission_workflow
    return repository, profiles, workflow


@router.post("/competition-profiles", response_model=CompetitionProfile)
def create_competition_profile(profile: CompetitionProfile, request: Request) -> CompetitionProfile:
    repository, profiles, _ = _services(request)
    profiles.register(profile)
    repository.persist_profile(profile)
    return profile


@router.get("/competition-profiles", response_model=list[CompetitionProfile])
def list_competition_profiles(request: Request) -> list[CompetitionProfile]:
    repository, profiles, _ = _services(request)
    persisted = {(item.profile_id, item.version): item for item in repository.list_profiles()}
    persisted.update({(item.profile_id, item.version): item for item in profiles.list()})
    return sorted(persisted.values(), key=lambda item: (item.name, item.version))


@router.get("/competition-profiles/{profile_id}", response_model=CompetitionProfile)
def get_competition_profile(
    profile_id: UUID, request: Request, version: int | None = None
) -> CompetitionProfile:
    repository, profiles, _ = _services(request)
    if version is not None:
        try:
            return profiles.get(profile_id, version)
        except KeyError:
            return repository.get_profile(profile_id, version)
    matching = [item for item in profiles.list() if item.profile_id == profile_id]
    if matching:
        return max(matching, key=lambda item: item.version)
    return repository.get_profile(profile_id)


async def _run(
    project_id: UUID,
    payload: FinalRunRequest,
    request: Request,
    *,
    force_freeze: bool | None = None,
) -> FinalRunResponse:
    repository, _, workflow = _services(request)
    outcome = await workflow.run(
        project_id,
        profile_id=payload.profile_id,
        profile_version=payload.profile_version,
        paper_id=payload.paper_id,
        paper_version=payload.paper_version,
        freeze_on_pass=(payload.freeze_on_pass if force_freeze is None else force_freeze),
    )
    package = outcome.package
    artifacts = (
        repository.list_artifacts(package.snapshot.submission_id) if package is not None else []
    )
    return FinalRunResponse(
        jury=outcome.jury,
        requirements=outcome.requirements,
        rules=outcome.rule_results,
        submission_check=outcome.check,
        snapshot=package.snapshot if package is not None else None,
        manifest=package.manifest if package is not None else None,
        artifacts=artifacts,
        state=outcome.state,
    )


@router.post("/projects/{project_id}/final/run", response_model=FinalRunResponse)
async def run_final(
    project_id: UUID, payload: FinalRunRequest, request: Request
) -> FinalRunResponse:
    return await _run(project_id, payload, request)


@router.post("/projects/{project_id}/final-jury", response_model=FinalRunResponse)
async def run_final_jury(
    project_id: UUID, payload: FinalRunRequest, request: Request
) -> FinalRunResponse:
    return await _run(project_id, payload, request, force_freeze=False)


@router.get("/projects/{project_id}/final-jury", response_model=FinalJuryReport)
def get_final_jury(project_id: UUID, request: Request) -> FinalJuryReport:
    repository, _, _ = _services(request)
    return repository.get_latest_jury(project_id)


@router.post("/projects/{project_id}/submission/check", response_model=SubmissionCheckResult)
async def check_submission(
    project_id: UUID, payload: FinalRunRequest, request: Request
) -> SubmissionCheckResult:
    _, _, workflow = _services(request)
    outcome = await workflow.run(
        project_id,
        profile_id=payload.profile_id,
        profile_version=payload.profile_version,
        paper_id=payload.paper_id,
        paper_version=payload.paper_version,
        freeze_on_pass=False,
    )
    return outcome.check


@router.get("/projects/{project_id}/submission/check", response_model=SubmissionCheckResult)
def get_submission_check(project_id: UUID, request: Request) -> SubmissionCheckResult:
    repository, _, _ = _services(request)
    return repository.get_latest_check(project_id)


@router.post("/projects/{project_id}/submission/freeze", response_model=FinalRunResponse)
async def freeze_submission(
    project_id: UUID, payload: FinalRunRequest, request: Request
) -> FinalRunResponse:
    return await _run(project_id, payload, request, force_freeze=True)


@router.post("/projects/{project_id}/submission/build", response_model=FinalRunResponse)
async def build_submission(
    project_id: UUID, payload: FinalRunRequest, request: Request
) -> FinalRunResponse:
    return await _run(project_id, payload, request, force_freeze=True)


@router.get("/projects/{project_id}/submission", response_model=SubmissionSnapshot)
def get_submission(project_id: UUID, request: Request) -> SubmissionSnapshot:
    repository, _, _ = _services(request)
    try:
        snapshot = repository.get_latest_snapshot(project_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=409, detail="submission snapshot failed integrity validation"
        ) from exc
    try:
        profile = repository.get_profile(
            snapshot.competition_profile_id, snapshot.competition_profile_version
        )
        artifacts = repository.list_artifacts(snapshot.submission_id)
        manifest = repository.get_manifest(snapshot.submission_id)
        manifest_artifact = next(item for item in artifacts if item.role.value == "MANIFEST")
        package_artifact = next(item for item in artifacts if item.role.value == "PACKAGE")
        protected = [item for item in artifacts if item.role.value not in {"MANIFEST", "PACKAGE"}]
        verifier: SubmissionIntegrityVerifier = request.app.state.submission_integrity_verifier
        package = SubmissionPackage(
            profile=profile,
            snapshot=snapshot,
            manifest=manifest,
            manifest_artifact=manifest_artifact,
            package_artifact=package_artifact,
            protected_artifacts=protected,
        )
        status = (
            verifier.verify(package)
            if repository.snapshot_context_valid(project_id, snapshot)
            else SubmissionStatus.DIRTY
        )
    except (StopIteration, ValueError):
        status = SubmissionStatus.DIRTY
    return (
        snapshot
        if status is SubmissionStatus.FROZEN
        else snapshot.model_copy(update={"status": status})
    )


@router.get("/projects/{project_id}/submission/artifacts", response_model=list[SubmissionArtifact])
def get_submission_artifacts(project_id: UUID, request: Request) -> list[SubmissionArtifact]:
    repository, _, _ = _services(request)
    snapshot = repository.get_latest_snapshot(project_id)
    return repository.list_artifacts(snapshot.submission_id)
