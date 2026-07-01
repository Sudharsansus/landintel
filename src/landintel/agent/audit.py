"""Natural-language audit trail per plot.

Single responsibility: turn what the deterministic and agent layers found about a
plot into one human-readable line, e.g.

    "Survey 100: closed, area 1.70 ha vs 1.67 stated (1.9%), 27 stones,
     43/55 measurements accepted — FLAGGED (3 issues)"

so the review queue and the client can see *why* a plot passed or was held. It
reads the plot (and, when given, the anomaly report) and emits prose; it makes no
decisions and changes nothing. The orchestrator appends the line to ``job.audit``.
"""

from __future__ import annotations

from ..core.enums import PlotStatus
from ..core.models import Plot
from .anomaly import AnomalyReport

__all__ = ["audit_plot"]

_VERDICT = {
    PlotStatus.VALIDATED: "OK",
    PlotStatus.FLAGGED: "FLAGGED",
    PlotStatus.FAILED: "FAILED",
}


def audit_plot(plot: Plot, *, report: AnomalyReport | None = None) -> str:
    """Return a one-line natural-language summary of a plot's state."""
    parts: list[str] = []

    boundary = plot.boundary
    if boundary is None or not boundary.points:
        parts.append("no boundary")
    elif boundary.is_closed:
        parts.append("closed")
        if plot.stated_area:
            computed_ha = boundary.computed_area / 10_000.0
            error = abs(computed_ha - plot.stated_area) / plot.stated_area
            parts.append(
                f"area {computed_ha:.2f} ha vs {plot.stated_area:.2f} stated ({error:.1%})"
            )
    else:
        parts.append(f"OPEN (gap {boundary.closure_gap:.1f} m)")

    accepted = sum(1 for m in plot.measurements if m.value is not None)
    parts.append(f"{accepted}/{len(plot.measurements)} measurements accepted")
    parts.append(f"{len(plot.corner_points)} stones")

    if report is not None and report.inconsistent_measurements:
        parts.append(
            f"{len(report.inconsistent_measurements)} measurement(s) inconsistent with edges"
        )

    verdict = _VERDICT.get(plot.status, plot.status.value.upper())
    if plot.flags:
        verdict += f" ({len(plot.flags)} issue{'s' if len(plot.flags) != 1 else ''})"

    return f"Survey {plot.survey_no}: " + ", ".join(parts) + f" — {verdict}"
