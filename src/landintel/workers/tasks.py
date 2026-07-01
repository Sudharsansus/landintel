"""Celery tasks — thin shell around the orchestrator.

Each task deserializes its arguments into domain objects, calls the orchestrator,
and translates domain errors into Celery failure signals. No pipeline logic lives
here; that all belongs in ``orchestrator.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .celery_app import celery_app
from ..core.models import Job
from ..pipeline.orchestrator import run_job

__all__ = ["run_job_task"]

logger = logging.getLogger(__name__)


def _persist_job(job: Job) -> None:
    """Save the updated job document to MongoDB using synchronous pymongo.

    Motor (async) cannot be used from a synchronous Celery task without an
    event-loop wrapper. pymongo is always available (motor depends on it) and
    the one-connection-per-job cost is acceptable here.
    """
    try:
        from pymongo import MongoClient
        from ..config import get_settings
        settings = get_settings()
        client: MongoClient = MongoClient(
            settings.mongo_uri, serverSelectionTimeoutMS=5000
        )
        try:
            col = client[settings.mongo_db]["jobs"]
            col.replace_one(
                {"client_id": job.client_id, "id": job.id},
                job.model_dump(mode="json"),
            )
        finally:
            client.close()
    except Exception:
        logger.exception("failed to persist job to DB", extra={"job_id": job.id})


@celery_app.task(
    bind=True,
    name="landintel.run_job",
    max_retries=0,       # orchestrator handles per-plot retries; job-level is once
    acks_late=True,
)
def run_job_task(self, job_dict: dict, output_dir: str) -> dict:
    """Execute a full pipeline job.

    Args:
        job_dict: JSON-serialized :class:`~landintel.core.models.Job`.
        output_dir: Directory path for intermediate outputs (DXFs).

    Returns:
        The updated job as a JSON-serializable dict.
    """
    job = Job.model_validate(job_dict)
    logger.info("task started", extra={"job_id": job.id})
    try:
        result = run_job(job, output_dir=Path(output_dir))
        _persist_job(result)
        return result.model_dump(mode="json")
    except Exception as exc:
        logger.exception("task failed unrecoverably", extra={"job_id": job.id})
        # Persist error state so the UI shows FAILED instead of stuck on QUEUED.
        job.error = f"{type(exc).__name__}: {exc}"
        _persist_job(job)
        raise self.retry(exc=exc, countdown=0, max_retries=0) from exc
