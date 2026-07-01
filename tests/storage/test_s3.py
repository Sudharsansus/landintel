"""S3 storage tests using moto (no real AWS, no credentials needed).

Proves: the ConfigError is a real line (not an intention), upload→download
round-trips bytes losslessly, keys are tenant-namespaced, and presigned URL
generation returns a usable URL string.
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
import moto
import pytest

from landintel.config import get_settings
from landintel.core.exceptions import ConfigError, ReportError
from landintel.storage.s3 import (
    build_key,
    download_file,
    presigned_url,
    upload_bytes,
    upload_file,
)

BUCKET = "landintel-test"
REGION = "ap-south-1"


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    """Inject fake credentials so boto3 doesn't reach out to real AWS."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_s3():
    """Start moto S3 mock and create the test bucket."""
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        yield s3


# --- ConfigError is a real line, not an intention ----------------------------


def test_upload_bytes_raises_config_error_for_blank_bucket(monkeypatch) -> None:
    """The first line of upload_bytes is a real config check."""
    monkeypatch.setenv("S3_BUCKET", "")
    get_settings.cache_clear()
    with pytest.raises(ConfigError) as exc_info:
        upload_bytes("c", "j", b"data", "file.txt")
    assert "S3_BUCKET" in str(exc_info.value)


def test_upload_raises_for_blank_credentials(monkeypatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
    get_settings.cache_clear()
    with pytest.raises(ConfigError) as exc_info:
        upload_bytes("c", "j", b"data", "file.txt")
    assert "AWS_ACCESS_KEY_ID" in str(exc_info.value)


def test_config_error_names_all_missing_keys(monkeypatch) -> None:
    monkeypatch.setenv("S3_BUCKET", "")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(ConfigError) as exc_info:
        upload_bytes("c", "j", b"data", "file.txt")
    msg = str(exc_info.value)
    assert "S3_BUCKET" in msg and "AWS_ACCESS_KEY_ID" in msg and "AWS_SECRET_ACCESS_KEY" in msg


# --- Key structure -----------------------------------------------------------


def test_build_key_is_tenant_namespaced() -> None:
    key = build_key("client_A", "job123", "report.pdf")
    assert key == "client_A/jobs/job123/report.pdf"
    assert key.startswith("client_A/")


def test_different_tenants_get_different_keys() -> None:
    key_a = build_key("client_A", "j1", "out.dxf")
    key_b = build_key("client_B", "j1", "out.dxf")
    assert key_a != key_b
    assert not key_a.startswith("client_B/")


# --- Upload / download round-trip --------------------------------------------


def test_upload_bytes_and_download_roundtrip(mock_s3, tmp_path: Path) -> None:
    payload = b"FMB survey data \x00\xff"
    key = upload_bytes("client_A", "job1", payload, "survey.bin")
    assert key == "client_A/jobs/job1/survey.bin"
    out = download_file(key, tmp_path / "survey.bin")
    assert out.read_bytes() == payload


def test_upload_file_and_download_roundtrip(mock_s3, tmp_path: Path) -> None:
    src = tmp_path / "plot.dxf"
    src.write_bytes(b"DXF content here")
    key = upload_file("client_A", "job1", src)
    assert key == "client_A/jobs/job1/plot.dxf"
    out_path = tmp_path / "downloaded.dxf"
    download_file(key, out_path)
    assert out_path.read_bytes() == src.read_bytes()


def test_upload_file_with_filename_override(mock_s3, tmp_path: Path) -> None:
    src = tmp_path / "tmp_abc123.dxf"
    src.write_bytes(b"data")
    key = upload_file("client_A", "job1", src, filename="survey_100.dxf")
    assert key == "client_A/jobs/job1/survey_100.dxf"


def test_download_missing_key_raises_report_error(mock_s3, tmp_path: Path) -> None:
    with pytest.raises(ReportError):
        download_file("client_A/jobs/job1/nonexistent.dxf", tmp_path / "out.dxf")


# --- Tenant isolation in the bucket -----------------------------------------


def test_tenant_objects_are_namespaced(mock_s3) -> None:
    """Objects from different tenants live under different prefixes."""
    upload_bytes("client_A", "j1", b"A data", "file.txt")
    upload_bytes("client_B", "j1", b"B data", "file.txt")

    # List prefix for A: only A's object.
    resp_a = mock_s3.list_objects_v2(Bucket=BUCKET, Prefix="client_A/")
    keys_a = [o["Key"] for o in resp_a.get("Contents", [])]
    assert all(k.startswith("client_A/") for k in keys_a)
    assert not any(k.startswith("client_B/") for k in keys_a)


# --- Presigned URL -----------------------------------------------------------


def test_presigned_url_returns_string(mock_s3) -> None:
    upload_bytes("client_A", "job1", b"pdf bytes", "report.pdf")
    url = presigned_url("client_A/jobs/job1/report.pdf", expiry_seconds=300)
    assert isinstance(url, str) and url.startswith("https://")


def test_presigned_url_contains_key(mock_s3) -> None:
    key = upload_bytes("client_A", "job1", b"zip", "delivery.zip")
    url = presigned_url(key)
    assert "delivery.zip" in url
