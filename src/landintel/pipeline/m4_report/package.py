"""Package job outputs and deliver via S3.

Single responsibility: zip the DXF outputs + PDF area statement + Excel sheet
into one delivery archive, upload it to S3 under the tenant-namespaced key path,
and return the presigned download URL.

Key path: ``{client_id}/jobs/{job_id}/delivery/landintel_delivery.zip``
This is consistent with the ``storage/s3.py`` convention and keeps all of a
tenant's job outputs under their prefix in the bucket.

All generation (PDF, Excel) happens before this module is called — this module
only assembles and ships.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from ...storage.s3 import presigned_url, upload_bytes
from ...core.models import Job
from .area_statement import generate_area_statement
from .excel_sheet import generate_excel_sheet

__all__ = ["package_and_deliver"]

DELIVERY_FILENAME = "landintel_delivery.zip"


def package_and_deliver(
    job: Job,
    *,
    expiry_seconds: int = 3600,
) -> str:
    """Build the delivery zip, upload to S3, and return a presigned URL.

    Args:
        job: The job whose outputs to package. ``job.output_files`` should
            contain the DXF paths written by M1/M2/M3.
        expiry_seconds: Presigned URL lifetime in seconds (default 1 hour).

    Returns:
        A presigned S3 URL the client can use to download the delivery zip.

    Raises:
        ConfigError: S3 is not configured (from ``storage.s3``).
        ReportError: Upload failed.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # DXF / DWG outputs from the pipeline stages.
        for path_str in job.output_files:
            path = Path(path_str)
            if path.exists():
                zf.write(path, arcname=path.name)

        # Area statement PDF.
        pdf_bytes = generate_area_statement(job)
        zf.writestr("area_statement.pdf", pdf_bytes)

        # Excel breakdown.
        xlsx_bytes = generate_excel_sheet(job)
        zf.writestr("area_breakdown.xlsx", xlsx_bytes)

    zip_bytes = buf.getvalue()
    key = upload_bytes(
        job.client_id,
        job.id,
        zip_bytes,
        DELIVERY_FILENAME,
        content_type="application/zip",
    )
    return presigned_url(key, expiry_seconds=expiry_seconds)
