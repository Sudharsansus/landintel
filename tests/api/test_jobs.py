"""API route tests using FastAPI TestClient + mongomock + Celery no-op.

No live Mongo, no Redis broker, no real S3. The lifespan is bypassed in tests
(no DB index creation, no real connection) — the routes are tested via
dependency injection with the mock db passed directly.

Tests exercise:
- Route logic and HTTP status codes
- Tenant isolation: client A cannot see client B's jobs via the HTTP surface
- deps.py wiring: current_client() flows into every repo call
- 404 for missing-or-wrong-tenant resources
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mongomock_motor
import pytest
from fastapi.testclient import TestClient

from landintel.api.deps import current_client, get_database
from landintel.api.main import app
from landintel.config import get_settings


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_db():
    return mongomock_motor.AsyncMongoMockClient()["landintel_test"]


def _make_client(mock_db, tenant: str) -> TestClient:
    """Build a TestClient with dependency overrides for the given tenant."""
    app.dependency_overrides[get_database] = lambda: mock_db
    app.dependency_overrides[current_client] = lambda: tenant
    # TestClient with lifespan disabled so no real Mongo connection is attempted.
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def client_a(mock_db):
    """TestClient scoped to client_A. Use alone — not with client_b in same test."""
    with patch("landintel.api.routes.jobs.run_job_task") as mock_task:
        mock_task.delay = MagicMock()
        with _make_client(mock_db, "client_A") as c:
            yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_b(mock_db):
    """TestClient scoped to client_B. Use alone — not with client_a in same test."""
    with patch("landintel.api.routes.jobs.run_job_task") as mock_task:
        mock_task.delay = MagicMock()
        with _make_client(mock_db, "client_B") as c:
            yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health() -> None:
    # Health has no DB dep — no override needed.
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------


def test_create_job_returns_202(client_a) -> None:
    r = client_a.post("/jobs", json={"input_files": ["client_A/jobs/j1/survey.pdf"]})
    assert r.status_code == 202
    body = r.json()
    assert body["client_id"] == "client_A"
    assert body["stage"] == "intake"
    assert "id" in body


def test_create_job_queues_celery_task(mock_db) -> None:
    app.dependency_overrides[get_database] = lambda: mock_db
    app.dependency_overrides[current_client] = lambda: "client_A"
    with patch("landintel.api.routes.jobs.run_job_task") as mock_task:
        mock_task.delay = MagicMock()
        with TestClient(app) as c:
            c.post("/jobs", json={"input_files": ["s3://bucket/file.pdf"]})
        assert mock_task.delay.called
    app.dependency_overrides.clear()


def test_get_job_own_client(client_a) -> None:
    create = client_a.post("/jobs", json={"input_files": ["file.pdf"]})
    job_id = create.json()["id"]
    r = client_a.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["id"] == job_id


def test_get_job_returns_404_for_missing(client_a) -> None:
    r = client_a.get("/jobs/nonexistent-id")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Tenant isolation through the HTTP surface
# ---------------------------------------------------------------------------
# Both clients share a mock_db but operate sequentially with explicit overrides
# to avoid the app.dependency_overrides race condition (last writer wins).

def _scoped_request(mock_db, tenant: str, method: str, path: str, **kwargs):
    """Make one request as ``tenant`` with a clean override, then restore."""
    with patch("landintel.api.routes.jobs.run_job_task") as mock_task:
        mock_task.delay = MagicMock()
        app.dependency_overrides[get_database] = lambda: mock_db
        app.dependency_overrides[current_client] = lambda: tenant
        try:
            with TestClient(app, raise_server_exceptions=True) as c:
                return getattr(c, method)(path, **kwargs)
        finally:
            app.dependency_overrides.clear()


def test_client_a_cannot_see_client_b_job(mock_db) -> None:
    """client_B's job is 404 when fetched as client_A — no data leak over HTTP."""
    create = _scoped_request(mock_db, "client_B", "post", "/jobs",
                             json={"input_files": ["b.pdf"]})
    job_id = create.json()["id"]

    r = _scoped_request(mock_db, "client_A", "get", f"/jobs/{job_id}")
    assert r.status_code == 404, (
        f"client_A received {r.status_code} for client_B's job — tenant leak!"
    )


def test_list_jobs_scoped_to_own_client(mock_db) -> None:
    """Each client's list shows only its own jobs."""
    _scoped_request(mock_db, "client_A", "post", "/jobs", json={"input_files": ["a1.pdf"]})
    _scoped_request(mock_db, "client_A", "post", "/jobs", json={"input_files": ["a2.pdf"]})
    _scoped_request(mock_db, "client_B", "post", "/jobs", json={"input_files": ["b1.pdf"]})

    list_a = _scoped_request(mock_db, "client_A", "get", "/jobs").json()["items"]
    list_b = _scoped_request(mock_db, "client_B", "get", "/jobs").json()["items"]

    assert len(list_a) == 2, f"client_A should see 2 jobs, got {len(list_a)}"
    assert len(list_b) == 1, f"client_B should see 1 job, got {len(list_b)}"
    assert all(j["client_id"] == "client_A" for j in list_a)
    assert all(j["client_id"] == "client_B" for j in list_b)


# ---------------------------------------------------------------------------
# List + pagination
# ---------------------------------------------------------------------------


def test_list_jobs_empty(client_a) -> None:
    r = client_a.get("/jobs")
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_list_jobs_pagination(client_a) -> None:
    for i in range(5):
        client_a.post("/jobs", json={"input_files": [f"file{i}.pdf"]})
    page1 = client_a.get("/jobs?limit=3&skip=0").json()["items"]
    page2 = client_a.get("/jobs?limit=3&skip=3").json()["items"]
    assert len(page1) == 3 and len(page2) == 2
    assert {j["id"] for j in page1}.isdisjoint({j["id"] for j in page2})


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_job_returns_204(client_a) -> None:
    create = client_a.post("/jobs", json={"input_files": ["f.pdf"]})
    job_id = create.json()["id"]
    r = client_a.delete(f"/jobs/{job_id}")
    assert r.status_code == 204


def test_cancel_wrong_client_returns_404(mock_db) -> None:
    create = _scoped_request(mock_db, "client_A", "post", "/jobs",
                             json={"input_files": ["f.pdf"]})
    job_id = create.json()["id"]
    r = _scoped_request(mock_db, "client_B", "delete", f"/jobs/{job_id}")
    assert r.status_code == 404
