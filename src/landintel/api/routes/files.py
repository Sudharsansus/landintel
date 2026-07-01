"""File upload — accept FMB PDFs and return S3 keys for job submission."""

from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile

from ...storage.s3 import upload_bytes
from ..deps import current_client

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload")
async def upload_fmb(
    file: UploadFile,
    job_id: str,
    client_id: str = Depends(current_client),
) -> dict:
    """Upload one FMB PDF; returns the S3 key to include in JobCreate.input_files."""
    data = await file.read()
    key = upload_bytes(client_id, job_id, data, file.filename or "upload.pdf",
                       content_type="application/pdf")
    return {"key": key, "filename": file.filename, "size": len(data)}
