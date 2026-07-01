"""S3 storage: upload, download, presigned URL.

Single responsibility: move bytes. No business logic about what gets stored —
the pipeline decides that, this module just moves it.

Use-time ConfigError contract: every public function calls ``_require_config()``
as its first action. If ``S3_BUCKET``, ``AWS_ACCESS_KEY_ID``, or
``AWS_SECRET_ACCESS_KEY`` is blank, a :class:`~landintel.core.exceptions.ConfigError`
is raised immediately naming the missing key(s) — not a cryptic boto3 error
40 frames deep when the actual upload attempt fails.

Key structure: ``{client_id}/jobs/{job_id}/{filename}``
Every object is namespaced by tenant so files can never collide across clients
and a per-tenant S3 lifecycle policy is trivially expressible as a prefix filter.
"""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from ..config import get_settings
from ..core.exceptions import ConfigError, ReportError

__all__ = [
    "build_key",
    "upload_file",
    "upload_bytes",
    "download_file",
    "presigned_url",
    "DEFAULT_EXPIRY_SECONDS",
]

DEFAULT_EXPIRY_SECONDS: int = 3600  # 1 hour


def _require_config() -> tuple[str, str, str, str]:
    """Return (bucket, key_id, secret, region) or raise ConfigError naming each blank.

    This is the first line of every public function. A blank value means the S3
    subsystem is not configured; raising here (not at import or startup) follows
    the same use-time pattern as ODA and Mongo.
    """
    settings = get_settings()
    missing = []
    if not settings.s3_bucket:
        missing.append("S3_BUCKET")
    if not settings.aws_access_key_id:
        missing.append("AWS_ACCESS_KEY_ID")
    if not settings.aws_secret_access_key:
        missing.append("AWS_SECRET_ACCESS_KEY")
    if missing:
        raise ConfigError(
            f"S3 storage is not configured; set in .env: {', '.join(missing)}",
            missing=missing,
        )
    return (
        settings.s3_bucket,
        settings.aws_access_key_id,
        settings.aws_secret_access_key,
        settings.aws_region,
    )


def _client(key_id: str, secret: str, region: str):
    return boto3.client(
        "s3",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name=region,
    )


def build_key(client_id: str, job_id: str, filename: str) -> str:
    """Canonical S3 key: ``{client_id}/jobs/{job_id}/{filename}``."""
    return f"{client_id}/jobs/{job_id}/{filename}"


def upload_file(
    client_id: str,
    job_id: str,
    local_path: Path | str,
    *,
    filename: str | None = None,
) -> str:
    """Upload a local file to S3 and return the object key.

    Args:
        client_id: Owning tenant — becomes the S3 key prefix.
        job_id: Job the file belongs to.
        local_path: The file to upload.
        filename: Override the key's filename part (default: the path's basename).

    Returns:
        The S3 object key.

    Raises:
        ConfigError: S3 credentials or bucket not configured.
        ReportError: The upload failed (boto3 ClientError).
    """
    bucket, key_id, secret, region = _require_config()
    path = Path(local_path)
    name = filename or path.name
    key = build_key(client_id, job_id, name)
    s3 = _client(key_id, secret, region)
    try:
        s3.upload_file(str(path), bucket, key)
    except ClientError as exc:
        raise ReportError(
            "S3 upload failed", key=key, reason=str(exc)
        ) from exc
    return key


def upload_bytes(
    client_id: str,
    job_id: str,
    data: bytes,
    filename: str,
    *,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload raw bytes to S3 and return the object key.

    Raises:
        ConfigError: S3 credentials or bucket not configured.
        ReportError: The upload failed.
    """
    bucket, key_id, secret, region = _require_config()
    key = build_key(client_id, job_id, filename)
    s3 = _client(key_id, secret, region)
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    except ClientError as exc:
        raise ReportError("S3 put_object failed", key=key, reason=str(exc)) from exc
    return key


def download_file(key: str, local_path: Path | str) -> Path:
    """Download an S3 object to ``local_path`` and return the path.

    Raises:
        ConfigError: S3 credentials or bucket not configured.
        ReportError: The download failed (key missing, permissions, network).
    """
    bucket, key_id, secret, region = _require_config()
    out = Path(local_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    s3 = _client(key_id, secret, region)
    try:
        s3.download_file(bucket, key, str(out))
    except ClientError as exc:
        raise ReportError("S3 download failed", key=key, reason=str(exc)) from exc
    return out


def presigned_url(key: str, *, expiry_seconds: int = DEFAULT_EXPIRY_SECONDS) -> str:
    """Generate a presigned GET URL for ``key``.

    The URL lets the browser download the file directly from S3 without exposing
    credentials. ``ResponseContentDisposition`` is set to ``attachment`` so the
    browser saves the file rather than trying to display it, and
    ``ResponseContentType`` is ``application/octet-stream`` to prevent MIME
    sniffing. Expiry defaults to 1 hour.

    Raises:
        ConfigError: S3 credentials or bucket not configured.
        ReportError: Presigned URL generation failed.
    """
    bucket, key_id, secret, region = _require_config()
    s3 = _client(key_id, secret, region)
    filename = Path(key).name
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ResponseContentType": "application/octet-stream",
                "ResponseContentDisposition": f'attachment; filename="{filename}"',
            },
            ExpiresIn=expiry_seconds,
        )
    except ClientError as exc:
        raise ReportError("presigned URL generation failed", key=key, reason=str(exc)) from exc
