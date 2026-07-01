"""Job repository: data access only, no business logic.

Every method takes ``client_id`` as its first required argument and injects it
into the filter. There is no ``find_job(job_id)`` that omits ``client_id`` —
the method signature makes cross-tenant leakage impossible to write accidentally.

Returns domain :class:`~landintel.core.models.Job` objects, not raw Mongo dicts.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...core.enums import JobStatus, Stage
from ...core.models import Job

__all__ = ["JobRepository"]


def _to_job(doc: dict) -> Job:
    doc.pop("_id", None)
    return Job.model_validate(doc)


class JobRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["jobs"]

    async def create(self, client_id: str, job: Job) -> Job:
        """Insert a new job; ``job.client_id`` must match ``client_id``."""
        assert job.client_id == client_id, "client_id mismatch on job create"
        doc = job.model_dump(mode="json")
        await self._col.insert_one(doc)
        return job

    async def get(self, client_id: str, job_id: str) -> Job | None:
        """Return the job if it belongs to this client, else None."""
        doc = await self._col.find_one({"client_id": client_id, "id": job_id})
        return _to_job(doc) if doc else None

    async def list(
        self,
        client_id: str,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> list[Job]:
        """List jobs for this client, newest first, optionally filtered by status."""
        filt: dict = {"client_id": client_id}
        if status is not None:
            filt["status"] = status.value
        cursor = self._col.find(filt).sort("created_at", -1).skip(skip).limit(limit)
        return [_to_job(doc) async for doc in cursor]

    async def update_stage(self, client_id: str, job_id: str, stage: Stage) -> bool:
        """Advance the job to a new stage. Returns True if the document was found."""
        result = await self._col.update_one(
            {"client_id": client_id, "id": job_id},
            {"$set": {"stage": stage.value}},
        )
        return result.matched_count == 1

    async def set_error(self, client_id: str, job_id: str, error: str) -> bool:
        """Record a fatal job-level error."""
        result = await self._col.update_one(
            {"client_id": client_id, "id": job_id},
            {"$set": {"error": error}},
        )
        return result.matched_count == 1

    async def append_audit(self, client_id: str, job_id: str, line: str) -> bool:
        """Append one audit-trail line."""
        result = await self._col.update_one(
            {"client_id": client_id, "id": job_id},
            {"$push": {"audit": line}},
        )
        return result.matched_count == 1

    async def update(self, client_id: str, job: Job) -> bool:
        """Replace the full job document. Used by the worker after pipeline completion."""
        doc = job.model_dump(mode="json")
        result = await self._col.replace_one(
            {"client_id": client_id, "id": job.id},
            doc,
        )
        return result.matched_count == 1
