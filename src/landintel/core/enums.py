"""Closed vocabularies shared across the whole system.

This module is the bottom of the dependency graph: it imports nothing from the
project (only the standard library) and everything else may import from it.

All enums derive from :class:`enum.StrEnum` so their members are real strings.
That makes them serialize transparently into JSON (API) and BSON (MongoDB)
without custom encoders, and lets repository queries compare against plain
string values stored in the database.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "JobStatus",
    "PlotStatus",
    "Stage",
    "LayerType",
    "MeasurementSource",
]


class JobStatus(StrEnum):
    """Lifecycle of a single conversion :class:`~landintel.core.models.Job`.

    A job moves forward through the happy path
    ``QUEUED -> RUNNING -> COMPLETED`` but can divert to ``NEEDS_REVIEW`` when
    the agent layer flags an anomaly that requires a human decision, or to
    ``FAILED`` when a module raises an unrecoverable error.
    """

    QUEUED = "queued"
    """Accepted and waiting for a Celery worker to pick it up."""

    RUNNING = "running"
    """A worker is actively executing the pipeline."""

    NEEDS_REVIEW = "needs_review"
    """Paused on an anomaly the agent could not resolve automatically."""

    COMPLETED = "completed"
    """All stages finished and outputs were delivered."""

    FAILED = "failed"
    """A stage raised an unrecoverable error; see the job audit trail."""

    CANCELLED = "cancelled"
    """Explicitly cancelled by an operator before completion."""


class PlotStatus(StrEnum):
    """Lifecycle of an individual survey :class:`~landintel.core.models.Plot`.

    One job may carry many plots (a village). Plots advance independently so a
    single flagged plot does not block the others from progressing.
    """

    EXTRACTED = "extracted"
    """Geometry, OCR numbers and metadata pulled from the FMB PDF (M1)."""

    VALIDATED = "validated"
    """OCR values passed the agent validator and area checks."""

    FLAGGED = "flagged"
    """An anomaly was detected; awaiting human review."""

    CORRECTED = "corrected"
    """A human edited one or more values; correction logged to memory."""

    GEOREFERENCED = "georeferenced"
    """Placed into real-world coordinates (M2)."""

    ASSEMBLED = "assembled"
    """Merged into the village map with boundaries snapped (M3)."""

    FAILED = "failed"
    """A hard, unrecoverable error on this plot (e.g. no boundary found).

    Distinct from ``FLAGGED``: a flagged plot is suspicious but processable and
    lets the job finish as ``NEEDS_REVIEW``; a failed plot blocks the job, which
    derives status ``JobStatus.FAILED``.
    """


class Stage(StrEnum):
    """The pipeline stage a job is currently in.

    Values mirror the four modules run in order by ``orchestrator.py``. Logged
    on every transition so the audit trail and dashboard can show progress.
    """

    INTAKE = "intake"
    """Input files received and registered, before M1 starts."""

    EXTRACT = "extract"
    """M1: FMB PDF -> structured DXF."""

    GEOREF = "georef"
    """M2: DXF -> georeferenced DWG."""

    ASSEMBLE = "assemble"
    """M3: N plot DXFs -> one village DWG."""

    REPORT = "report"
    """M4: reports generated and package delivered to S3."""

    DELIVERED = "delivered"
    """Terminal stage: presigned URL available to the client."""


class LayerType(StrEnum):
    """Canonical DXF layer names produced by M1.

    Values match the Sivagangai fixture DXFs from the client exactly — verified
    against ``tests/fixtures/DXF/SIVAGANGAI_Manamadurai_TPudukkottai_100.dxf``.
    Written verbatim as layer names by ``to_dxf.py``; colors and linetypes are
    also applied there to match the client's visual standard.
    """

    DEFAULT = "0"
    """The mandatory AutoCAD default layer."""

    DEFPOINTS = "Defpoints"
    """Non-plotting construction points (AutoCAD convention, mixed case)."""

    BOUNDARY = "BOUNDARY"
    """Outer plot boundary ring — Red (ACI 1), Continuous."""

    SUBDIVISION_LINES = "SUBDIVISION_LINES"
    """Sub-plot division geometry — Green (ACI 3), Continuous."""

    SUBDIVISION = "SUBDIVISION"
    """Sub-plot text labels — Teal (ACI 130)."""

    CHAIN_LINES = "CHAIN_LINES"
    """Survey chain / traverse lines — CHAINLINE linetype, dark gray (ACI 251)."""

    BLUE_STROKES = "BLUE_STROKES"
    """Blue filled markers: chain arrows and direction indicators."""

    CHAINLINE_DIMENSIONS = "CHAINLINE_DIMENSIONS"
    """Measured lengths annotated along chain lines — color 251."""

    BOUNDARY_DIMENSIONS = "BOUNDARY_DIMENSIONS"
    """Measured lengths annotated along boundary segments — color 220."""

    DIMENSIONS = "DIMENSIONS"
    """Sub-division dimension annotations — color 200."""

    SEPARATION_LINE = "SEPARATION_LINE"
    """Divider ticks — White/Black (ACI 7)."""

    STONES = "STONES"
    """Corner survey stones (A, B, C, ...): TEXT labels — White/Black (ACI 7)."""

    SURVEY_NUMBER = "SURVEY_NUMBER"
    """The survey number text label — Red (ACI 1)."""

    WELL_AND_BUILDING = "WELL and BUILDING"
    """Wells and permanent structures — Blue (ACI 5)."""

    DASHED_REF = "DASHED_REF"
    """Gray dashed neighbor-boundary reference ticks — White/Black (ACI 7), DASHED linetype."""

    NEIGHBOR_LABEL = "neighbor label"
    """Neighbor survey-number labels placed at the shared boundary — White (ACI 7)."""


class MeasurementSource(StrEnum):
    """Provenance of a :class:`~landintel.core.models.Measurement` value.

    Drives trust and auditing: an ``OCR`` value may be wrong and is subject to
    validation, whereas a ``CORRECTED`` value was set by a human and is treated
    as ground truth (and logged to the corrections collection for memory).
    """

    OCR = "ocr"
    """Read directly from the rasterized FMB image by PaddleOCR."""

    CORRECTED = "corrected"
    """Overridden by a human reviewer; authoritative."""
