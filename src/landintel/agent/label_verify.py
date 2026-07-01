"""Per-measurement label verification -- the trust score the deliverable renders on.

This is the layer that attacks the "map labels don't match the manual DWG" problem
HEAD-ON, within the honest limits of OCR recall. It does NOT touch geometry (the
boundary is always the vector ring) -- it scores each *dimension label* so the
deliverable can show the trusted ones confidently and mark the rest provisional,
instead of asserting a wrong number on the map.

Two independent signals, strongest first:

1. **External ground truth** (optional): when the client supplies the exact edge
   lengths for a survey (from a LandXML / CSV / surveyor file), a read value that
   matches an exact edge to within ``GROUND_TRUTH_TOL_M`` is CONFIRMED (1.0), and
   the exact value is offered as the correction. This is the real fix -- it
   replaces a noisy OCR label with the surveyed number.

2. **Geometry self-consistency**: the read value vs the real-world length of the
   line it was anchored to (``Measurement.line_length_m``). Agreement within
   ``CONSISTENCY_TOLERANCE`` is corroborating evidence (the same signal anomaly.py
   already REPORTS); disagreement means the token is probably a mis-anchored
   neighbour/sub-plot number, not an edge measurement.

The score is written to ``Measurement.label_confidence`` in place. It is a DISPLAY
/ correction signal, never a hard gate -- consistent with the documented reality
that ~half of anchored numeric tokens are non-measurements, so gating would flood
review. ``to_dxf`` / M3 annotation can read it to style provisional labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.models import Measurement, Plot

__all__ = [
    "LabelVerification",
    "LabelReport",
    "verify_labels",
    "inconsistent_measurements",
    "CONSISTENCY_TOLERANCE",
    "GROUND_TRUTH_TOL_M",
]

CONSISTENCY_TOLERANCE = 0.15
"""Relative value-vs-edge error above which a label is counted inconsistent.

Kept equal to ``anomaly.CONSISTENCY_TOLERANCE`` so the two layers agree on what
"inconsistent" means; anomaly.py now sources its diagnostic from here."""

GROUND_TRUTH_TOL_M = 0.30
"""Absolute metres within which a read value is treated as confirming an exact
(surveyed) edge length -- survey-grade, so a tight tolerance."""


@dataclass(frozen=True)
class LabelVerification:
    """The verification outcome for one measurement."""

    raw: str
    value: float | None
    status: str          # "confirmed" | "consistent" | "inconsistent" | "unanchored"
    confidence: float    # written to Measurement.label_confidence
    corrected_value: float | None = None
    """The exact edge length to use instead, when ground truth confirmed/corrected it."""


@dataclass
class LabelReport:
    """Aggregate of a verify_labels pass over one plot."""

    verifications: list[LabelVerification] = field(default_factory=list)
    confirmed: int = 0          # matched an external exact measurement
    consistent: int = 0         # agreed with its own anchored edge
    inconsistent: int = 0       # disagreed with its anchored edge (likely non-measurement)
    unanchored: int = 0         # no edge to check against
    corrections: int = 0        # values replaced by an exact ground-truth length

    @property
    def inconsistent_raws(self) -> list[str]:
        return [v.raw for v in self.verifications if v.status == "inconsistent"]

    @property
    def trusted_fraction(self) -> float:
        """Fraction of valued measurements that are confirmed or self-consistent."""
        valued = [v for v in self.verifications if v.value is not None]
        if not valued:
            return 0.0
        good = sum(1 for v in valued if v.status in ("confirmed", "consistent"))
        return good / len(valued)


def _nearest_truth(value: float, truths: list[float]) -> tuple[float, float] | None:
    """Closest exact edge length to ``value`` and the absolute gap, or None."""
    if not truths:
        return None
    best = min(truths, key=lambda t: abs(t - value))
    return best, abs(best - value)


def verify_labels(
    plot: Plot,
    ground_truth_edges: list[float] | None = None,
) -> LabelReport:
    """Score every measurement on ``plot`` and write ``label_confidence`` in place.

    Parameters
    ----------
    plot : the validated plot (measurements may have ``value`` set by validator.py).
    ground_truth_edges : optional list of EXACT edge lengths (m) for this survey,
        e.g. from a LandXML/CSV/surveyor source. When given, a read value matching
        one within ``GROUND_TRUTH_TOL_M`` is confirmed and corrected to the exact
        value. Order-independent (nearest match), since OCR labels are not ordered.

    Returns
    -------
    LabelReport with per-measurement verifications and aggregate counts. Geometry
    is never modified; only ``Measurement.label_confidence`` (and, when a ground
    truth confirms it, the report's ``corrected_value``) are set.
    """
    truths = list(ground_truth_edges or [])
    report = LabelReport()

    for m in plot.measurements:
        v = _verify_one(m, truths)
        m.label_confidence = v.confidence
        report.verifications.append(v)
        if v.status == "confirmed":
            report.confirmed += 1
            if v.corrected_value is not None:
                report.corrections += 1
        elif v.status == "consistent":
            report.consistent += 1
        elif v.status == "inconsistent":
            report.inconsistent += 1
        else:
            report.unanchored += 1

    return report


def _verify_one(m: Measurement, truths: list[float]) -> LabelVerification:
    value = m.value
    if value is None:
        # Unparseable token -- no claim to make; lowest trust.
        return LabelVerification(m.raw, None, "unanchored", 0.0)

    # Signal 1: external exact measurement (strongest).
    nt = _nearest_truth(value, truths)
    if nt is not None and nt[1] <= GROUND_TRUTH_TOL_M:
        return LabelVerification(m.raw, value, "confirmed", 1.0, corrected_value=nt[0])

    # Signal 2: geometry self-consistency vs its anchored edge.
    edge = m.line_length_m
    if edge:
        rel = abs(value - edge) / edge
        if rel <= CONSISTENCY_TOLERANCE:
            # Confidence tapers from 1.0 (exact) down to ~0.5 at the tolerance edge.
            conf = 1.0 - 0.5 * (rel / CONSISTENCY_TOLERANCE)
            return LabelVerification(m.raw, value, "consistent", round(conf, 3))
        return LabelVerification(m.raw, value, "inconsistent", 0.2)

    # Valued but nothing to check it against.
    return LabelVerification(m.raw, value, "unanchored", 0.5)


def inconsistent_measurements(
    plot: Plot, ground_truth_edges: list[float] | None = None
) -> list[str]:
    """Raws whose value disagrees with their anchored edge (and no ground truth).

    The single source of truth for the value-vs-edge diagnostic that anomaly.py
    surfaces -- computed here so the LABEL trust score and the anomaly diagnostic
    can never drift apart."""
    return verify_labels(plot, ground_truth_edges).inconsistent_raws
