"""DB repository tests using mongomock-motor (no live Mongo required).

Exercises real query logic — filters, sorts, pagination, upsert — including the
central correctness property: a query scoped to client A cannot return client B's
document. That test is the anchor for the whole tenant-isolation guarantee.

All tests are async (pytest-asyncio auto mode, set in pyproject.toml).
"""

from __future__ import annotations

import pytest
import mongomock_motor

from landintel.core.enums import JobStatus, PlotStatus, Stage
from landintel.core.models import Boundary, Correction, Job, Plot
from landintel.db.repositories.corrections import CorrectionRepository
from landintel.db.repositories.jobs import JobRepository
from landintel.db.repositories.plots import PlotRepository


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def mock_db():
    """A fresh in-memory Mongo database per test."""
    client = mongomock_motor.AsyncMongoMockClient()
    return client["landintel_test"]


@pytest.fixture
def job_repo(mock_db):
    return JobRepository(mock_db)


@pytest.fixture
def plot_repo(mock_db):
    return PlotRepository(mock_db)


@pytest.fixture
def correction_repo(mock_db):
    return CorrectionRepository(mock_db)


def make_job(client_id: str, stage: Stage = Stage.INTAKE) -> Job:
    return Job(client_id=client_id, stage=stage)


def make_plot(client_id: str, survey_no: str = "100") -> Plot:
    return Plot(
        client_id=client_id,
        survey_no=survey_no,
        district="D", taluk="T", village="V",
        boundary=Boundary(
            points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]
        ),
    )


def make_correction(client_id: str, plot_id: str = "100") -> Correction:
    return Correction(
        client_id=client_id,
        job_id="job1",
        plot_id=plot_id,
        field="measurement",
        old="98",
        new="40.6",
    )


# ============================================================================
# TENANT ISOLATION — the anchor test
# ============================================================================


async def test_tenant_isolation_jobs(job_repo: JobRepository) -> None:
    """A query scoped to client A can never return client B's job."""
    job_a = make_job("client_A")
    job_b = make_job("client_B")
    await job_repo.create("client_A", job_a)
    await job_repo.create("client_B", job_b)

    # Client A's query returns only A's job.
    jobs_for_a = await job_repo.list("client_A")
    assert len(jobs_for_a) == 1
    assert jobs_for_a[0].client_id == "client_A"
    assert jobs_for_a[0].id == job_a.id

    # Client B's query returns only B's job.
    jobs_for_b = await job_repo.list("client_B")
    assert len(jobs_for_b) == 1
    assert jobs_for_b[0].id == job_b.id

    # Fetching B's ID as A returns nothing.
    leaked = await job_repo.get("client_A", job_b.id)
    assert leaked is None


async def test_tenant_isolation_plots(plot_repo: PlotRepository) -> None:
    """Client A cannot retrieve client B's plot even with the same survey_no."""
    plot_a = make_plot("client_A", "100")
    plot_b = make_plot("client_B", "100")  # same survey_no, different tenant
    await plot_repo.upsert("client_A", plot_a)
    await plot_repo.upsert("client_B", plot_b)

    # A's scoped query returns A's document.
    got_a = await plot_repo.get("client_A", "100")
    assert got_a is not None and got_a.client_id == "client_A"

    # B's scoped query returns B's document.
    got_b = await plot_repo.get("client_B", "100")
    assert got_b is not None and got_b.client_id == "client_B"

    # The two documents are distinct (not the same object leaked across tenants).
    assert got_a.client_id != got_b.client_id


async def test_tenant_isolation_corrections(correction_repo: CorrectionRepository) -> None:
    """Corrections from client A never appear in client B's feed."""
    corr_a = make_correction("client_A")
    corr_b = make_correction("client_B")
    await correction_repo.record("client_A", corr_a)
    await correction_repo.record("client_B", corr_b)

    feed_a = await correction_repo.list_recent("client_A")
    assert all(c.client_id == "client_A" for c in feed_a)
    assert len(feed_a) == 1

    feed_b = await correction_repo.list_recent("client_B")
    assert all(c.client_id == "client_B" for c in feed_b)


# ============================================================================
# JobRepository
# ============================================================================


async def test_create_and_get_job(job_repo: JobRepository) -> None:
    job = make_job("client_A")
    created = await job_repo.create("client_A", job)
    fetched = await job_repo.get("client_A", job.id)
    assert fetched is not None
    assert fetched.id == job.id and fetched.client_id == "client_A"
    assert isinstance(fetched, Job)


async def test_get_missing_job_returns_none(job_repo: JobRepository) -> None:
    result = await job_repo.get("client_A", "nonexistent")
    assert result is None


async def test_list_jobs_newest_first(job_repo: JobRepository) -> None:
    import asyncio
    j1 = make_job("c")
    await asyncio.sleep(0.01)
    j2 = make_job("c")
    await job_repo.create("c", j1)
    await job_repo.create("c", j2)
    jobs = await job_repo.list("c")
    assert jobs[0].id == j2.id  # newest first


async def test_list_jobs_pagination(job_repo: JobRepository) -> None:
    for _ in range(5):
        await job_repo.create("c", make_job("c"))
    page1 = await job_repo.list("c", limit=3, skip=0)
    page2 = await job_repo.list("c", limit=3, skip=3)
    assert len(page1) == 3 and len(page2) == 2
    ids1 = {j.id for j in page1}
    ids2 = {j.id for j in page2}
    assert ids1.isdisjoint(ids2)


async def test_update_stage(job_repo: JobRepository) -> None:
    job = make_job("c")
    await job_repo.create("c", job)
    ok = await job_repo.update_stage("c", job.id, Stage.EXTRACT)
    assert ok
    fetched = await job_repo.get("c", job.id)
    assert fetched.stage is Stage.EXTRACT
    # status is derived; EXTRACT with plots -> RUNNING
    assert fetched.status is JobStatus.RUNNING


async def test_update_stage_wrong_client_returns_false(job_repo: JobRepository) -> None:
    job = make_job("c")
    await job_repo.create("c", job)
    ok = await job_repo.update_stage("other", job.id, Stage.EXTRACT)
    assert not ok


async def test_append_audit(job_repo: JobRepository) -> None:
    job = make_job("c")
    await job_repo.create("c", job)
    await job_repo.append_audit("c", job.id, "line one")
    await job_repo.append_audit("c", job.id, "line two")
    fetched = await job_repo.get("c", job.id)
    assert fetched.audit == ["line one", "line two"]


async def test_set_error(job_repo: JobRepository) -> None:
    job = make_job("c")
    await job_repo.create("c", job)
    await job_repo.set_error("c", job.id, "something went wrong")
    fetched = await job_repo.get("c", job.id)
    assert fetched.error == "something went wrong"
    assert fetched.status is JobStatus.FAILED


# ============================================================================
# PlotRepository
# ============================================================================


async def test_upsert_and_get_plot(plot_repo: PlotRepository) -> None:
    plot = make_plot("c", "42")
    await plot_repo.upsert("c", plot)
    fetched = await plot_repo.get("c", "42")
    assert fetched is not None and fetched.survey_no == "42"
    assert isinstance(fetched, Plot)


async def test_upsert_replaces_existing(plot_repo: PlotRepository) -> None:
    plot = make_plot("c", "42")
    await plot_repo.upsert("c", plot)
    plot.flags.append("first flag")
    await plot_repo.upsert("c", plot)
    fetched = await plot_repo.get("c", "42")
    # Upsert replaced: only one document exists.
    all_plots = await plot_repo.list_by_status("c", PlotStatus.EXTRACTED)
    assert len(all_plots) == 1


async def test_update_plot_status(plot_repo: PlotRepository) -> None:
    plot = make_plot("c", "42")
    await plot_repo.upsert("c", plot)
    ok = await plot_repo.update_status("c", "42", PlotStatus.VALIDATED)
    assert ok
    fetched = await plot_repo.get("c", "42")
    assert fetched.status is PlotStatus.VALIDATED


async def test_list_plots_by_status(plot_repo: PlotRepository) -> None:
    await plot_repo.upsert("c", make_plot("c", "1"))
    await plot_repo.upsert("c", make_plot("c", "2"))
    await plot_repo.update_status("c", "2", PlotStatus.VALIDATED)
    extracted = await plot_repo.list_by_status("c", PlotStatus.EXTRACTED)
    assert len(extracted) == 1 and extracted[0].survey_no == "1"


# ============================================================================
# CorrectionRepository
# ============================================================================


async def test_record_and_list_corrections(correction_repo: CorrectionRepository) -> None:
    c1 = make_correction("c", "100")
    c2 = make_correction("c", "101")
    await correction_repo.record("c", c1)
    await correction_repo.record("c", c2)
    all_corr = await correction_repo.list_recent("c")
    assert len(all_corr) == 2
    assert all(isinstance(c, Correction) for c in all_corr)


async def test_list_corrections_filtered_by_plot(correction_repo: CorrectionRepository) -> None:
    await correction_repo.record("c", make_correction("c", "100"))
    await correction_repo.record("c", make_correction("c", "101"))
    only_100 = await correction_repo.list_recent("c", plot_id="100")
    assert len(only_100) == 1 and only_100[0].plot_id == "100"


async def test_corrections_newest_first(correction_repo: CorrectionRepository) -> None:
    import asyncio
    c1 = make_correction("c")
    await asyncio.sleep(0.01)
    c2 = make_correction("c")
    await correction_repo.record("c", c1)
    await correction_repo.record("c", c2)
    feed = await correction_repo.list_recent("c")
    assert feed[0].id == c2.id  # newest first
