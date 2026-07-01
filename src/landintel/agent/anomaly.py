"""Judge a validated Plot at the geometry level: flag, fail, or pass.

Single responsibility: read the numbers ``build_plot`` and ``validator`` produced
and decide, at the *plot* level, whether the geometry is trustworthy. It gates on
things that are reliable signals of a good or broken plot:

* boundary closes (an open ring is unreliable -> flag, never crash);
* computed polygon area matches the FMB stated area (this is where the area
  cross-check, set up but deliberately un-judged in ``build_plot``, becomes the
  active judge);
* there are enough corner stones to define the plot.

The flag-vs-fail distinction (``ValidationFlag`` vs ``GeometryError`` from
``core.exceptions``) fires here as ``PlotStatus.FLAGGED`` vs ``FAILED``: a
suspicious-but-recoverable plot is flagged and the village keeps moving; only a
catastrophically wrong one fails and blocks the job.

What it does NOT gate on: per-measurement value-vs-edge-length consistency. That
genuinely catches a mis-anchored number, but on real fixtures ~half of accepted
measurements are inconsistent (≈half of anchored numeric tokens are
non-measurements), so gating would flood review with false positives. It is
therefore REPORTED as a diagnostic (``inconsistent_measurements``) for the review
UI, not turned into a flag. Output *geometry* is guaranteed correct regardless;
the measurement labels are noisy by the nature of the OCR recall, and that is a
known, documented limitation -- not something this layer can honestly gate on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.enums import PlotStatus
from ..core.models import Plot

__all__ = [
    "AnomalyIssue",
    "AnomalyReport",
    "check_plot",
    "AREA_FLAG_TOLERANCE",
    "AREA_FAIL_TOLERANCE",
    "MIN_CORNER_STONES",
]

# Area tolerances (relative). Calibrated against the fixtures: the clean plots
# land at 1.9% (survey 100), 5.1% (199) and 2.0% (31), well inside the flag band.
AREA_FLAG_TOLERANCE = 0.10
"""Above this relative area error the plot is flagged for review."""

AREA_FAIL_TOLERANCE = 0.50
"""Above this the plot is a hard failure -- the geometry or scale is fundamentally
wrong, not merely off."""

MIN_CORNER_STONES = 3
"""Fewer corner stones than this cannot define a plot polygon."""

CONSISTENCY_TOLERANCE = 0.15
"""A measurement whose value differs from its edge length by more than this is
counted as inconsistent (diagnostic only)."""


@dataclass(frozen=True)
class AnomalyIssue:
    """One plot-level finding. ``severity`` is ``"flag"`` or ``"fail"``."""

    code: str
    severity: str
    message: str


@dataclass
class AnomalyReport:
    """The outcome of judging one plot."""

    issues: list[AnomalyIssue] = field(default_factory=list)
    inconsistent_measurements: list[str] = field(default_factory=list)
    """Diagnostic: raws whose value disagrees with their edge length. Not a gate."""

    @property
    def ok(self) -> bool:
        """True when no flag/fail issue was found (diagnostics do not count)."""
        return not self.issues

    @property
    def failed(self) -> bool:
        return any(issue.severity == "fail" for issue in self.issues)


def _check_boundary(plot: Plot, issues: list[AnomalyIssue]) -> bool:
    """Return whether a usable closed boundary exists; record issues found."""
    boundary = plot.boundary
    if boundary is None or not boundary.points:
        issues.append(AnomalyIssue("no_boundary", "fail", "no boundary geometry extracted"))
        return False
    if not boundary.is_closed:
        issues.append(
            AnomalyIssue(
                "non_closing",
                "flag",
                f"boundary does not close (gap {boundary.closure_gap:.1f} m)",
            )
        )
        return False
    return True


def _check_area(plot: Plot, issues: list[AnomalyIssue]) -> None:
    """Compare computed polygon area to the FMB stated area."""
    if plot.stated_area is None or plot.stated_area <= 0:
        issues.append(AnomalyIssue("no_stated_area", "flag", "no stated area to verify against"))
        return
    computed_ha = plot.boundary.computed_area / 10_000.0  # type: ignore[union-attr]
    error = abs(computed_ha - plot.stated_area) / plot.stated_area
    message = (
        f"computed {computed_ha:.3f} ha vs stated {plot.stated_area:.3f} ha "
        f"({error:.1%})"
    )
    if error > AREA_FAIL_TOLERANCE:
        issues.append(AnomalyIssue("area_mismatch", "fail", message))
    elif error > AREA_FLAG_TOLERANCE:
        issues.append(AnomalyIssue("area_mismatch", "flag", message))


def _check_stones(plot: Plot, issues: list[AnomalyIssue]) -> None:
    if len(plot.corner_points) < MIN_CORNER_STONES:
        issues.append(
            AnomalyIssue(
                "too_few_stones",
                "flag",
                f"only {len(plot.corner_points)} corner stones (need {MIN_CORNER_STONES})",
            )
        )


def _inconsistent_measurements(plot: Plot) -> list[str]:
    """Diagnostic: accepted measurements whose value disagrees with their edge.

    Delegates to ``agent.label_verify`` (the single source of truth for the
    value-vs-edge check) so the per-label trust score the deliverable renders and
    this plot-level diagnostic can never drift apart. As a side effect it tags each
    measurement's ``label_confidence``."""
    from .label_verify import inconsistent_measurements
    return inconsistent_measurements(plot)


def check_plot(plot: Plot) -> AnomalyReport:
    """Judge ``plot`` at the geometry level, mutating its flags and status.

    Appends a reason to ``plot.flags`` for each issue and sets ``plot.status`` to
    ``FAILED`` if any issue is a hard failure, else ``FLAGGED`` if any issue was
    found. A clean plot's status is left as the validator set it (it never
    downgrades a validator flag).
    """
    issues: list[AnomalyIssue] = []

    has_boundary = _check_boundary(plot, issues)
    if has_boundary:
        _check_area(plot, issues)
    _check_stones(plot, issues)

    report = AnomalyReport(
        issues=issues,
        inconsistent_measurements=_inconsistent_measurements(plot),
    )

    for issue in issues:
        plot.flags.append(f"[{issue.code}] {issue.message}")
    if report.failed:
        plot.status = PlotStatus.FAILED
    elif issues:
        plot.status = PlotStatus.FLAGGED

    return report
