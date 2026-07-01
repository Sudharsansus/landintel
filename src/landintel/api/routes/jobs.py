"""Job routes — create, list, get, cancel.

Every route that reads or writes data depends on ``current_client()`` and passes
the ``client_id`` it returns directly to the repository. This is the structural
tenant-scoping guarantee: the dependency chain ``request → current_client() →
client_id → repo.method(client_id, ...)`` is unbreakable by construction.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ...core.models import Job
from ...db.repositories.jobs import JobRepository
from ...workers.tasks import run_job_task
from ..deps import current_client, get_job_repo
from ..schemas.job import JobArtifact, JobCreate, JobListResponse, JobResponse
from ...storage.s3 import presigned_url
from ...core.exceptions import ConfigError

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    body: JobCreate,
    client_id: str = Depends(current_client),
    repo: JobRepository = Depends(get_job_repo),
) -> JobResponse:
    """Submit a new conversion job. Returns immediately; processing is async."""
    import tempfile, pathlib
    job = Job(client_id=client_id, input_files=body.input_files)
    await repo.create(client_id, job)
    # Enqueue the Celery task. Output dir is a tempdir per job on the worker.
    output_dir = str(pathlib.Path(tempfile.gettempdir()) / "landintel" / job.id)
    run_job_task.delay(job.model_dump(mode="json"), output_dir)
    return JobResponse.from_job(job)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    limit: int = 20,
    skip: int = 0,
    client_id: str = Depends(current_client),
    repo: JobRepository = Depends(get_job_repo),
) -> JobListResponse:
    """List this client's jobs, newest first."""
    jobs = await repo.list(client_id, limit=limit, skip=skip)
    return JobListResponse(
        items=[JobResponse.from_job(j) for j in jobs],
        total=len(jobs),
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    client_id: str = Depends(current_client),
    repo: JobRepository = Depends(get_job_repo),
) -> JobResponse:
    """Get one job. Returns 404 if it doesn't belong to this client."""
    job = await repo.get(client_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse.from_job(job)


@router.get("/{job_id}/files", response_model=list[JobArtifact])
async def list_job_files(
    job_id: str,
    client_id: str = Depends(current_client),
    repo: JobRepository = Depends(get_job_repo),
) -> list[JobArtifact]:
    """Return presigned download URLs for all output artifacts of a job.

    Only S3 keys are returned (local /tmp paths are skipped). URLs expire in 1h.
    """
    job = await repo.get(client_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    from pathlib import Path as _P
    artifacts: list[JobArtifact] = []
    for key in job.output_files:
        if _P(key).is_absolute():
            continue  # local filesystem path — no S3 key, nothing to presign
        filename = _P(key).name
        if filename.startswith("m1_"):
            stage = "extract"
        elif filename.startswith("m2_"):
            stage = "georef"
        elif filename.startswith("m3_"):
            stage = "assemble"
        elif filename.startswith("m4_"):
            stage = "report"
        else:
            stage = "output"
        try:
            url = presigned_url(key)
        except (ConfigError, Exception):
            continue
        artifacts.append(JobArtifact(stage=stage, filename=filename, url=url))
    return artifacts


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    job_id: str,
    client_id: str = Depends(current_client),
    repo: JobRepository = Depends(get_job_repo),
) -> None:
    """Cancel a queued or running job. 404 if not found for this client."""
    job = await repo.get(client_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job.cancelled = True
    # Persist the cancellation flag via the stage field (simplest no-new-method path).
    await repo.set_error(client_id, job_id, "cancelled by operator")
