"""Domain models passed between pipeline stages.

These are plain data objects (pydantic v2) with no I/O: no DB, no FastAPI, no
Celery. Modules take them in and return them out; only the orchestrator knows
the full sequence.

Design rules enforced here:

* **Tenant-ready.** Every persisted root document (:class:`Plot`, :class:`Job`,
  :class:`Correction`) carries ``client_id``.
* **No drifting derived state.** Facts that can be computed from other fields --
  whether a boundary closes, a polygon's area, a job's overall status -- are
  exposed as ``@property`` and are therefore *not* stored. They recompute on
  access and can never fall out of sync with the data they derive from. In
  particular ``Job.status`` is derived from the job's stage and its plots, not a
  mutable field an operator can set wrong.
* **Lean.** Every field is read by some module. Speculative fields are omitted
  until the module that needs them is built.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enums import JobStatus, MeasurementSource, PlotStatus, Stage

__all__ = [
    "Point",
    "CornerPoint",
    "Measurement",
    "Boundary",
    "SubPlot",
    "SubPlotLabel",
    "NeighborLabel",
    "Plot",
    "Job",
    "Correction",
]

# A 2D coordinate in drawing units (DXF model space). Real-world coordinates
# only exist after M2 georeferencing; until then everything is drawing units.
Point = tuple[float, float]

# Two endpoints closer than this (drawing units) are treated as the same point
# for the purpose of "is this ring geometrically closed as drawn". Survey-level
# tolerance ("does this boundary close *well enough*") is a policy decision the
# anomaly layer makes by thresholding ``Boundary.closure_gap`` itself.
CLOSURE_TOLERANCE = 1e-6


def _utcnow() -> datetime:
    """Timezone-aware UTC now (used as a default factory)."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """A fresh opaque identifier."""
    return uuid4().hex


class CornerPoint(BaseModel):
    """A labelled survey corner stone (A, B, C, ...) on the ``STONES`` layer.

    Doubles as a georeferencing anchor candidate in M2 when paired with GPS.

    ``x, y`` is the stone's true coordinate — snapped to the connecting point
    (the line junction it marks).  This is both what M2 uses for stone-matching
    and the LEFT-justified text insertion point in the DXF, so the number's grip
    in AutoCAD sits exactly on the line-endpoint node.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    x: float
    y: float


class Measurement(BaseModel):
    """One measured value read off the FMB image and anchored to a line.

    Keeps the full provenance triple plus confidence so downstream layers have
    unambiguous signals:

    * ``raw`` -- exactly what OCR emitted, comma decimals and all ("44,2").
      Preserved verbatim so the corrections feedback loop can learn what OCR
      *saw* versus what it *should* have been.
    * ``value`` -- the normalized numeric value (``44.2``), or ``None`` when the
      raw text could not be parsed into a plausible number.
    * ``source`` -- OCR or a human correction (see :class:`MeasurementSource`).
    * ``confidence`` -- OCR confidence in ``[0, 1]``. This is the field the
      anomaly layer thresholds to decide flag-vs-pass: a value present at low
      confidence is a :class:`~landintel.core.exceptions.ValidationFlag`,
      whereas OCR returning nothing at all is a hard ``OCRFailure``.
    """

    model_config = ConfigDict(extra="forbid")

    raw: str
    value: float | None = None
    source: MeasurementSource = MeasurementSource.OCR
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    line_ref: str | None = None
    """Identifier of the line/segment this number annotates (set by anchor.py)."""

    line_class: str | None = None
    """Feature class of the labelled line: ``boundary`` / ``internal`` / ``chain``.

    Routes the dimension to its DXF layer in ``to_dxf.py`` (boundary dimensions to
    ``BOUNDARY_DIMENSIONS``, chain to ``CHAINLINE_DIMENSIONS``, etc.).
    """

    position: Point | None = None
    """Where the number sits on the sheet, in the plot's real-world metre frame.

    Set by ``build_plot`` (the OCR detection centre, transformed). ``to_dxf``
    places the dimension text here; ``None`` when unknown.
    """

    line_angle: float | None = None
    """Angle of the labelled line in DXF coordinate space (degrees, [0, 180)).

    Computed by ``build_plot`` from the anchored segment after the y-axis flip
    (PDF y-down → DXF y-up).  Used by ``to_dxf`` to rotate the TEXT entity so
    the label runs parallel to the line it annotates.  ``None`` for chain
    dimension labels that are not anchored to a specific segment.
    """

    line_length_m: float | None = None
    """Real-world length (m) of the line this measurement labels.

    Set by ``build_plot`` from the anchored segment. The value *should* equal
    this length; ``anomaly.py`` reports a large disagreement as a per-measurement
    consistency diagnostic (it is a diagnostic, not a gate -- ~half of anchored
    numeric tokens are non-measurements, so gating would flood review).
    """

    label_confidence: float | None = None
    """Independent trust score in ``[0, 1]`` for this measurement as a LABEL.

    Set by ``agent/label_verify.py`` (not OCR confidence): how well the read value
    agrees with the geometry/edge it annotates (and an external exact measurement
    when one is supplied). This drives which dimension labels the deliverable
    renders confidently vs marks provisional -- it does NOT move geometry, which is
    always from the vector boundary. ``None`` until a verification pass runs.
    """


class Boundary(BaseModel):
    """An ordered ring of points describing a plot outline.

    Closure and area are **derived**, never stored, so they cannot go stale when
    ``points`` is edited. The anomaly layer reads :attr:`closure_gap` (the actual
    distance between the first and last point) to apply its own survey tolerance,
    and :attr:`is_closed` for the simple boolean check.
    """

    model_config = ConfigDict(extra="forbid")

    points: list[Point] = Field(default_factory=list)

    @property
    def closure_gap(self) -> float:
        """Distance between the first and last point (drawing units).

        ``inf`` when there are too few points to form a ring, so an incomplete
        boundary never reads as closed.
        """
        if len(self.points) < 3:
            return math.inf
        (x0, y0), (xn, yn) = self.points[0], self.points[-1]
        return math.hypot(xn - x0, yn - y0)

    @property
    def is_closed(self) -> bool:
        """Whether the ring is geometrically closed within ``CLOSURE_TOLERANCE``."""
        return self.closure_gap <= CLOSURE_TOLERANCE

    @property
    def computed_area(self) -> float:
        """Polygon area via the shoelace formula, in drawing units squared.

        The ring is treated as implicitly closed (last point joined to first).
        This is geometric area in *drawing* units; converting to real-world area
        (hectares) for the FMB-stated-area check happens after M2 applies the
        scale. Returns ``0.0`` for degenerate rings (fewer than 3 points).
        """
        pts = self.points
        if len(pts) < 3:
            return 0.0
        total = 0.0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0


class SubPlotLabel(BaseModel):
    """A sub-plot label token (2A, 3B, 5B, …) read by OCR from the FMB image.

    Placed on the ``SUB DIVISION NUMBER`` layer at the OCR detection position
    (real-world metres).  Distinct from :class:`SubPlot` which carries full
    boundary geometry; this is the lightweight label-only representation produced
    by M1 before sub-plot ring resolution.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    position: Point


class NeighborLabel(BaseModel):
    """A neighbor survey-number label (e.g. 99, 101, 102) found near a shared boundary.

    Populated by ``build_plot`` from anchor-layer neighbor detections.
    Written to the ``neighbor label`` (ACI 7) layer by ``to_dxf``.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    position: Point


class SubPlot(BaseModel):
    """A subdivision within a plot (the ``SUBDIVISION`` / ``SUBDIVISION_LINES``).

    Carries its own boundary so M3 can place the sub-plot label and area at the
    sub-polygon's centroid. ``boundary`` is optional because a subdivision may be
    detected by label before its ring is resolved.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    boundary: Boundary | None = None


class Plot(BaseModel):
    """One survey number: the unit M1 extracts and M2/M3 place into the village.

    Tracks its own :class:`PlotStatus` and advances independently of sibling
    plots, so a single flagged or failed plot does not stall the rest of the job.
    When the anomaly layer flags a plot it appends a human-readable reason to
    :attr:`flags`, which the review UI surfaces alongside the ``FLAGGED`` status.
    """

    model_config = ConfigDict(extra="forbid")

    client_id: str
    survey_no: str
    """Natural key of the plot within a job; also used as ``Correction.plot_id``."""

    district: str
    taluk: str
    village: str
    stated_area: float | None = None
    """FMB header-stated area (hectares), the ground truth the area check uses.

    ``None`` when the header could not be parsed.
    """

    scale: int | None = None
    """FMB drawing-scale denominator from the header (e.g. 2021 for "1 : 2021").

    Per-PDF. Applied in ``build_plot.py`` to convert pixel geometry to real-world
    metres; ``None`` when the header scale could not be parsed.
    """

    boundary: Boundary | None = None
    sub_plots: list[SubPlot] = Field(default_factory=list)
    sub_plot_labels: list[SubPlotLabel] = Field(default_factory=list)
    """Sub-division label tokens (2A, 3B, 5B, …) at their OCR positions.

    Populated by ``build_plot`` from the anchor layer's sub-plot detections.
    Written to the ``SUB DIVISION NUMBER`` (teal, ACI 130) layer by ``to_dxf``.
    """
    measurements: list[Measurement] = Field(default_factory=list)
    corner_points: list[CornerPoint] = Field(default_factory=list)

    subdivision_segments: list[tuple[Point, Point]] = Field(default_factory=list)
    """Internal sub-division line segments in real-world metres, (start, end) pairs.

    Populated by ``build_plot`` from ``PageVectors.internal``. Written to the
    ``SUB DIVISION`` (green) layer by ``to_dxf``. Empty until build_plot runs.
    """

    chain_segments: list[tuple[Point, Point]] = Field(default_factory=list)
    """Chain/traverse line segments in real-world metres, (start, end) pairs.

    Populated by ``build_plot`` from ``PageVectors.chain``. Written to the
    ``chain line`` (ACAD_ISO06W100) layer by ``to_dxf``. Empty until build_plot runs.
    """

    separation_segments: list[tuple[Point, Point]] = Field(default_factory=list)
    """Separation tick segments in real-world metres, (start, end) pairs.

    Populated by ``build_plot`` from ``PageVectors.separation``. Written to the
    ``S.F.SEPERATION LINE`` (DASHDOT) layer by ``to_dxf``. Empty until build_plot runs.
    """

    dashed_ref_segments: list[tuple[Point, Point]] = Field(default_factory=list)
    """Gray dashed neighbor-boundary reference ticks in real-world metres, (start, end) pairs.

    Populated by ``build_plot`` from ``PageVectors.dashed_ref``. Written to the
    ``DASHED_REF`` (DASHED) layer by ``to_dxf``. Empty until build_plot runs.
    """

    neighbor_labels: list[NeighborLabel] = Field(default_factory=list)
    """Neighbor survey-number labels (99, 101, 102, …) near shared boundaries.

    Populated by ``build_plot`` from anchor-layer neighbor detections.
    Written to ``neighbor label`` (ACI 7) by ``to_dxf``.
    """

    survey_glyph_center: Point | None = None
    """Real-world-metre centre of the largest blue survey-number glyph on the page.

    Set by ``build_plot`` when the glyph extraction sentinel detection is present.
    ``to_dxf`` writes the survey number at this position (in addition to the
    boundary centroid) so it lands where the surveyor physically drew it.
    """

    status: PlotStatus = PlotStatus.EXTRACTED
    flags: list[str] = Field(default_factory=list)
    """Reasons this plot was flagged for review (one per anomaly)."""


class Job(BaseModel):
    """A single conversion run: N input FMB PDFs -> village outputs.

    ``status`` is **derived** from ``stage``, the per-plot statuses, and the two
    externally-driven control fields (``cancelled``, ``error``). Nothing sets
    ``status`` directly, so it can never disagree with the plots: a job with a
    hard-failed plot reads ``FAILED``; a finished job with any flagged plot reads
    ``NEEDS_REVIEW``; otherwise a finished job reads ``COMPLETED``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    client_id: str
    stage: Stage = Stage.INTAKE
    plots: list[Plot] = Field(default_factory=list)
    input_files: list[str] = Field(default_factory=list)
    output_files: list[str] = Field(default_factory=list)
    audit: list[str] = Field(default_factory=list)
    """Natural-language audit trail, one entry per significant event (audit.py)."""

    cancelled: bool = False
    error: str | None = None
    """Set only on a job-level fatal error not attributable to a single plot."""

    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def status(self) -> JobStatus:
        """Overall job status, derived so it cannot drift from reality.

        Precedence: an explicit cancel or fatal error dominates; then a
        hard-failed plot blocks the whole job; then the lifecycle position
        (queued before work starts, running mid-pipeline) and finally, once
        delivered, review-vs-complete based on whether any plot is flagged.
        """
        if self.cancelled:
            return JobStatus.CANCELLED
        if self.error is not None:
            return JobStatus.FAILED
        if any(plot.status is PlotStatus.FAILED for plot in self.plots):
            return JobStatus.FAILED
        if self.stage is Stage.INTAKE:
            return JobStatus.QUEUED
        if self.stage is Stage.DELIVERED:
            if any(plot.status is PlotStatus.FLAGGED for plot in self.plots):
                return JobStatus.NEEDS_REVIEW
            return JobStatus.COMPLETED
        return JobStatus.RUNNING


class Correction(BaseModel):
    """A human override of a single value, logged from day one for the feedback loop.

    Linked to the *specific* measurement it corrects (via ``measurement_ref``,
    matching ``Measurement.line_ref``), not merely to the plot, so the eventual
    retrieval logic can ask "for this kind of measurement, what did OCR read and
    what was it actually?". ``old_source`` records whether the value being
    overridden came from OCR or was itself a prior correction.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    client_id: str
    job_id: str
    plot_id: str
    """The corrected plot's ``survey_no``."""

    field: str
    """Which field was corrected, e.g. ``"measurement"`` or ``"stated_area"``."""

    measurement_ref: str | None = None
    """``Measurement.line_ref`` when correcting a measurement; ``None`` otherwise."""

    old: str
    new: str
    old_source: MeasurementSource = MeasurementSource.OCR
    created_at: datetime = Field(default_factory=_utcnow)
