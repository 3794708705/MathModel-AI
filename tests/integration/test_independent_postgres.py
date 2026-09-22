import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mathmodel_ai.verification.independent_repository import IndependentVerificationRepository
from tests.integration.test_independent_verification import (
    create_verification_app,
    prepare_formal_plan,
)

DATABASE_URL = os.getenv("MM_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
    pytest.mark.skipif(DATABASE_URL is None, reason="MM_TEST_DATABASE_URL is not configured"),
]


def test_postgres_real_replay_restart_readback_and_concurrency(tmp_path):
    app = create_verification_app(tmp_path, DATABASE_URL)
    with TestClient(app) as client:
        plan, _, _ = prepare_formal_plan(app, client)
        service = app.state.independent_verification
        service.recompute(plan.attempt_id)
        view = service.replay(plan.attempt_id, "demand_stress")
        assert view.status.value == "PASS", view
        # Dispose only this test's pool; then reopen through a separate engine/session factory.
        engine = create_engine(DATABASE_URL)
        factory = sessionmaker(engine, expire_on_commit=False)
        fresh = IndependentVerificationRepository(factory)
        assert fresh.get(plan.attempt_id)[0] == plan
        assert len(fresh.jobs(plan.plan_id)) == 2
        probe = (
            "import os,sys; from uuid import UUID; "
            "from sqlalchemy import create_engine; "
            "from sqlalchemy.orm import sessionmaker; "
            "from mathmodel_ai.verification.independent_repository "
            "import IndependentVerificationRepository; "
            "engine=create_engine(os.environ['MM_TEST_DATABASE_URL']); "
            "r=IndependentVerificationRepository(sessionmaker(engine)); "
            "p,_=r.get(UUID(sys.argv[1])); "
            "assert len(r.jobs(p.plan_id))==2; print('RESTART_READBACK_PASS')"
        )
        restarted = subprocess.run(
            [
                sys.executable,
                "-c",
                probe,
                str(plan.attempt_id),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert restarted.returncode == 0, restarted.stderr
        assert restarted.stdout.strip() == "RESTART_READBACK_PASS"
        with ThreadPoolExecutor(max_workers=4) as pool:
            claims = list(
                pool.map(lambda _: fresh.claim(plan.plan_id, "concurrency-probe"), range(8))
            )
        winners = [j for j in claims if j is not None]
        assert len(winners) == 1
        fresh.finish(winners[0], {"error": "TEST_ONLY_NOT_AN_ACCEPTANCE_SCENARIO"})
        service.repository = fresh
        assert service.view(plan.attempt_id) == view
        assert service.replay(plan.attempt_id, "demand_stress") == view
        engine.dispose()
