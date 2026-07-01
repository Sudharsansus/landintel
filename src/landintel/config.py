"""The single source of truth for configuration.

Nothing else in the codebase reads ``os.environ`` directly -- every value comes
from the :class:`Settings` object returned by :func:`get_settings`. Settings are
loaded from the process environment and an optional ``.env`` file (see
``.env.example`` for the full, documented set of keys).

Fail-fast contract: if a required key is missing, :func:`get_settings` raises
:class:`~landintel.core.exceptions.ConfigError` naming the offending keys,
rather than letting the app boot half-configured and crash several modules deep.

Settings are grouped logically (app, Mongo, Redis, S3, Anthropic, OCR, ODA).
Required secrets have no default; everything else defaults to a sensible local
value so the system runs out of the box for development.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from .core.exceptions import ConfigError

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """All runtime configuration, validated once at startup.

    Field names map to upper-cased environment variables (``client_id`` ->
    ``CLIENT_ID``). Required fields (no default) must be present or
    :func:`get_settings` raises :class:`ConfigError`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App / tenancy -------------------------------------------------------
    environment: Literal["local", "production"] = "local"
    client_id: str = "client_default"
    """The single hardcoded tenant for now; ``deps.current_client()`` returns it."""

    log_level: str = "INFO"

    # --- MongoDB -------------------------------------------------------------
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "landintel"

    # --- Redis (Celery broker + result backend) ------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- AWS S3 (delivery) ---------------------------------------------------
    # Optional at startup so M1-only development needs no AWS; storage/s3.py
    # raises ConfigError at use-time if these are still blank when delivery runs.
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-south-1"
    s3_bucket: str = ""

    # --- Anthropic (the agent brain) -----------------------------------------
    anthropic_api_key: str  # required: the agent layer cannot run without it
    anthropic_model: str = "claude-opus-4-8"

    # --- OCR engine selection ------------------------------------------------
    ocr_engine: str = "paddle"
    """Which OCR engine to use: ``paddle`` (PaddleOCR, default) or ``vision``
    (Google Cloud Vision API).  Set ``OCR_ENGINE=vision`` in production once
    GOOGLE_CREDENTIALS_JSON is configured in the Render dashboard."""

    # --- PaddleOCR PP-OCRv5 (used when ocr_engine = "paddle") ---------------
    # Mobile det/rec locally (low RAM); flip ocr_det_model to the server det in
    # production for higher accuracy -- without touching ocr.py.
    ocr_det_model: str = "PP-OCRv5_mobile_det"
    ocr_rec_model: str = "en_PP-OCRv5_mobile_rec"
    ocr_zoom: float = 3.0
    """Page render zoom before OCR (3x ~= 216-300 DPI), per the sample-data findings."""

    ocr_use_mkldnn: bool = False
    """Disable oneDNN/mkldnn by default; some host CPUs crash with it enabled."""

    # --- Google Cloud Vision (used when ocr_engine = "vision") ---------------
    google_credentials_json: str = ""
    """Service-account JSON *content* (not a file path).  When set, the code
    writes this to a temp file and points GOOGLE_APPLICATION_CREDENTIALS at it
    so the Vision client authenticates via ADC.  If blank, standard ADC applies
    (GOOGLE_APPLICATION_CREDENTIALS env var pointing to a file, or the GCP
    metadata server when running on GCP).  Set in the Render dashboard with
    sync: false — never commit real credentials."""

    # --- ODA File Converter (DXF -> DWG, M2) ---------------------------------
    oda_converter_path: str = ""
    """Absolute path to the ODA File Converter binary; checked at use-time in M2."""

    # --- GIS / coordinate systems (pyproj, offline -- no API key) ------------
    gis_source_crs: str = "EPSG:4326"
    """Input CRS: WGS84, the GPS lat/lng corner anchors come in as (M2)."""

    gis_target_crs: str = "EPSG:32644"
    """Projected CRS: UTM Zone 44N, covers Tamil Nadu (metres)."""

    gis_snap_tolerance_m: float = 0.5
    """Shared-boundary snap threshold in metres (M3)."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, loaded and validated once.

    Cached so configuration is parsed a single time. Raises
    :class:`ConfigError` -- naming every missing required key -- instead of
    surfacing a raw pydantic ``ValidationError``, so a misconfiguration is a
    clear startup failure rather than an obscure stack trace.
    """
    try:
        return Settings()  # type: ignore[call-arg]  # values come from env/.env
    except ValidationError as exc:
        missing = sorted(
            ".".join(str(part) for part in err["loc"])
            for err in exc.errors()
            if err["type"] == "missing"
        )
        if missing:
            raise ConfigError(
                "Missing required configuration: "
                + ", ".join(missing)
                + ". Set them in .env or the environment (see .env.example).",
                missing=missing,
            ) from exc
        raise ConfigError("Invalid configuration.", errors=exc.errors()) from exc
