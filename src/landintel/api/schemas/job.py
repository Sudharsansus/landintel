"""API schemas for jobs — flat, JSON-serializable, API-contract stable.

Separate from :class:`~landintel.core.models.Job` deliberately: the domain model
carries computed properties (``status``) and mutable state that can't round-trip
through JSON cleanly. These schemas are what the client sees and what we
guarantee to keep stable across releases.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from ...core.enums import JobStatus, Stage

__all__ = ["JobCreate", "JobResponse", "JobListResponse", "JobArtifact"]


class JobCreate(BaseModel):
    """Request body for POST /jobs — the client submits a list of PDF S3 keys."""
    input_files: list[str]


class JobResponse(BaseModel):
    """One job as returned by the API."""
    id: str
    client_id: str
    status: JobStatus
    stage: Stage
    input_files: list[str]
    output_files: list[str]
    audit: list[str]
    created_at: datetime

    @classmethod
    def from_job(cls, job) -> "JobResponse":
        return cls(
            id=job.id,
            client_id=job.client_id,
            status=job.status,   # derived property — computed here, not stored
            stage=job.stage,
            input_files=job.input_files,
            output_files=job.output_files,
            audit=job.audit,
            created_at=job.created_at,
        )


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int


class JobArtifact(BaseModel):
    """One downloadable output file from a completed pipeline stage."""
    stage: str       # e.g. "extract", "georef", "report"
    filename: str    # just the basename, e.g. "m1_extract_survey_405.dxf"
    url: str         # presigned S3 URL, valid for 1 hour
