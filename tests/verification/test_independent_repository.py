from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import IndependentVerificationJobRecord
from mathmodel_ai.schemas.independent_verification import MetricSpec, ScenarioSpec, VerificationPlan
from mathmodel_ai.verification.independent_repository import IndependentVerificationRepository
from mathmodel_ai.verification.metric_recompute import content_digest


@pytest.fixture
def repository(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'jobs.sqlite'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    repository = IndependentVerificationRepository(factory)
    metric = MetricSpec(metric_id="rmse", key="rmse", version="1")
    plan = VerificationPlan(
        attempt_id=uuid4(),
        result_id=uuid4(),
        version="1",
        source_artifact_id=uuid4(),
        source_sha256="a" * 64,
        metrics=[metric],
        scenarios=[ScenarioSpec(scenario_id="stress", version="1", metrics=[metric])],
        scientific_scope="Schema/persistence fixture; not numerical acceptance.",
    )
    repository.register(plan, "b" * 64)
    return repository, plan, factory


def test_idempotency_concurrent_claim_and_terminal_immutability(repository):
    repo, plan, _ = repository
    repo.register(plan, "b" * 64)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = list(pool.map(lambda _: repo.claim(plan.plan_id, "baseline"), range(8)))
    unique = [j for j in jobs if j is not None]
    assert len(unique) == 1
    assert repo.jobs(plan.plan_id) == {"baseline": None}
    repo.finish(unique[0], {"batch": {"metrics": []}})
    with pytest.raises(ValueError, match="immutable"):
        repo.finish(unique[0], {})
    with pytest.raises(ValueError, match="immutable"):
        repo.register(plan.model_copy(update={"source_sha256": "f" * 64}), "b" * 64)


def test_job_status_and_payload_tamper_never_trusted(repository):
    repo, plan, factory = repository
    job_id = repo.claim(plan.plan_id, "baseline")
    repo.finish(job_id, {"batch": {"metrics": []}})
    with factory() as session:
        row = session.scalar(select(IndependentVerificationJobRecord))
        row.payload_json = {"status": "PASS"}
        session.commit()
    with pytest.raises(ValueError, match="integrity"):
        repo.jobs(plan.plan_id)
    with factory() as session:
        row = session.get(IndependentVerificationJobRecord, job_id)
        row.payload_digest = content_digest(row.payload_json)
        row.status = "RUNNING"
        session.commit()
    with pytest.raises(ValueError, match="terminal outcome"):
        repo.jobs(plan.plan_id)
