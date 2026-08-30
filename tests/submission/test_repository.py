from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select

from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import CompetitionProfileRecord, CompetitionRuleRecord, Project
from mathmodel_ai.db.session import create_session_factory, session_scope
from mathmodel_ai.schemas.submission import CorrectionPlan, CorrectionScope, RuleSeverity
from mathmodel_ai.submission.profiles import generic_modeling_test_profile
from mathmodel_ai.submission.repository import SubmissionRepository


def test_profile_versions_and_correction_plans_are_immutable() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    repository = SubmissionRepository(factory)
    profile_v1 = generic_modeling_test_profile()
    profile_v2 = profile_v1.model_copy(update={"version": 2, "name": "fixture-v2"})
    repository.persist_profile(profile_v1)
    repository.persist_profile(profile_v2)
    assert repository.get_profile(profile_v1.profile_id, 1) == profile_v1
    assert repository.get_profile(profile_v1.profile_id) == profile_v2
    with pytest.raises(ValueError, match="immutable competition profile"):
        repository.persist_profile(profile_v1.model_copy(update={"name": "tampered"}))
    with session_scope(factory) as session:
        row = session.scalar(
            select(CompetitionProfileRecord).where(
                CompetitionProfileRecord.profile_id == profile_v1.profile_id,
                CompetitionProfileRecord.version == 1,
            )
        )
        assert row is not None
        row.profile_json = {**row.profile_json, "name": "persisted tamper"}
    with pytest.raises(ValueError, match="COMPETITION_PROFILE_INTEGRITY_ERROR"):
        repository.get_profile(profile_v1.profile_id, 1)
    with session_scope(factory) as session:
        profile_row = session.scalar(
            select(CompetitionProfileRecord).where(
                CompetitionProfileRecord.profile_id == profile_v2.profile_id,
                CompetitionProfileRecord.version == 2,
            )
        )
        assert profile_row is not None
        rule_row = session.scalar(
            select(CompetitionRuleRecord).where(
                CompetitionRuleRecord.profile_record_id == profile_row.id
            )
        )
        assert rule_row is not None
        rule_row.rule_json = {**rule_row.rule_json, "description": "tampered rule"}
    with pytest.raises(ValueError, match="rule records disagree"):
        repository.get_profile(profile_v2.profile_id, 2)

    project_id = uuid4()
    with session_scope(factory) as session:
        session.add(Project(id=project_id, name="Correction persistence"))
    plan = CorrectionPlan(
        project_id=project_id,
        finding_refs=["JURY-format"],
        scope=CorrectionScope.FORMAT_ONLY,
        affected_components=["paper filename"],
        requires_model_change=False,
        requires_result_change=False,
        requires_paper_change=True,
        required_revalidation=["PAPER", "FINAL_JURY", "SUBMISSION"],
        risk=RuleSeverity.MINOR,
        priority=1,
    )
    repository.persist_correction(plan)
    repository.persist_correction(plan)
    assert repository.list_corrections(project_id) == [plan]
    with pytest.raises(ValueError, match="immutable correction plan"):
        repository.persist_correction(plan.model_copy(update={"priority": 2}))
    engine.dispose()
