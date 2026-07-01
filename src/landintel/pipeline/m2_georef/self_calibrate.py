"""Self-calibrating ACCEPT gate -- adapt the chain-coverage bar to each span.

The static gate (``CHAIN_COVER_ACCEPT = 0.50``) was calibrated on INGUR, where true
matches land at 58-100% coverage and coincidental ones at 12-43%, so 0.50 sits in
the clean gap. A NEW district can have a different gap: if its coincidental matches
are stronger (say up to 55%), a fixed 0.50 would let one through. This module reads
the ACTUAL per-span distribution of chain-coverage values and, when the data shows a
clear bimodal gap, raises the ACCEPT bar to the middle of that gap.

CRITICAL 0-FP invariant: calibration may ONLY TIGHTEN. The returned threshold is
``max(floor, gap_midpoint)`` clamped to ``[floor, ceil]`` -- it can move the bar UP
(reject more) but never below the validated floor (which would risk a false accept).
With too few samples or no clear gap it returns the floor unchanged, so behaviour on
INGUR and every already-validated span is identical. The pipeline applies the result
as a DEMOTE-ONLY pass (ACCEPT -> REVIEW), never a promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

_log = logging.getLogger(__name__)

MIN_SAMPLES = 5
"""Below this many candidate coverages, do not calibrate (return the floor)."""

MIN_GAP = 0.10
"""A bimodal valley narrower than this is not trustworthy -> return the floor."""

DANGER_MARGIN = 0.15
"""Only treat a valley as the coincidental/true boundary when the cluster BELOW it
tops out within this margin of the floor (i.e. a coincidental cluster that crept just
above the floor). A high gap inside an all-real spread (e.g. 0.85->1.0) is NOT a
coincidental boundary, so it must never raise the bar and demote real matches."""

EPS = 0.02
"""Threshold within this of the floor counts as 'at the floor' -> unchanged."""


@dataclass(frozen=True)
class Calibration:
    threshold: float
    floor: float
    ceil: float
    n_samples: int
    gap_lo: float = 0.0
    gap_hi: float = 0.0
    calibrated: bool = False
    reason: str = ""


def calibrate_coverage_threshold(
    coverages: list[float],
    floor: float = 0.50,
    ceil: float = 0.95,
    min_samples: int = MIN_SAMPLES,
    min_gap: float = MIN_GAP,
    danger_margin: float = DANGER_MARGIN,
) -> Calibration:
    """Derive a span-specific ACCEPT chain-coverage threshold (tighten-only).

    Pass the FULL candidate distribution (NO_COVERAGE + REVIEW + ACCEPT), so the
    low coincidental cluster is visible. Among the empty bands ("valleys") between
    sorted coverages, take the LOWEST one whose lower cluster tops out within
    ``danger_margin`` of the floor -- i.e. the boundary just above a coincidental
    cluster that crept above the floor -- and set the bar at its midpoint.

    Choosing the LOWEST such valley (not the widest gap) is what protects real
    matches: on INGUR the valley sits at ~0.50 between the 0.43 coincidental top and
    the 0.58 lowest true match, so the bar stays ~floor and the 0.58 match is kept.
    A new district whose coincidental matches reach 0.55 yields a valley at ~0.64,
    raising the bar to reject them. Returns the floor unchanged when there are too
    few samples or no qualifying valley (e.g. an all-real spread).

    Parameters
    ----------
    coverages : every candidate plot's chain_coverage in the span (0..1).
    floor : the validated static ACCEPT bar; the result never goes below it.
    ceil : never raise the bar above this (true matches can sit at ~0.6-1.0; we
        must not demand near-total coverage or we would reject corridor-clipped
        real plots).
    danger_margin : only a valley whose lower side is <= ``floor + danger_margin``
        is treated as a coincidental/true boundary.
    """
    vals = sorted(v for v in coverages if v == v)  # drop NaN
    n = len(vals)
    base = Calibration(threshold=floor, floor=floor, ceil=ceil, n_samples=n)
    if n < min_samples:
        return Calibration(**{**base.__dict__, "reason": f"only {n} samples (<{min_samples})"})

    # LOWEST valley with width >= min_gap, midpoint in [floor, ceil], and lower
    # side within danger_margin of the floor (a coincidental cluster near the bar).
    chosen: tuple[float, float] | None = None
    for a, b in zip(vals, vals[1:]):
        mid = 0.5 * (a + b)
        if (b - a) >= min_gap and floor <= mid <= ceil and a <= floor + danger_margin:
            chosen = (a, b)
            break  # vals is sorted -> first qualifying valley is the lowest

    if chosen is None:
        return Calibration(**{**base.__dict__,
                              "reason": "no coincidental-cluster valley near floor"})

    best_lo, best_hi = chosen
    midpoint = min(max(0.5 * (best_lo + best_hi), floor), ceil)
    calibrated = (midpoint - floor) > EPS
    threshold = midpoint if calibrated else floor
    cal = Calibration(
        threshold=round(threshold, 4), floor=floor, ceil=ceil, n_samples=n,
        gap_lo=round(best_lo, 4), gap_hi=round(best_hi, 4),
        calibrated=calibrated,
        reason=(f"valley [{best_lo:.2f}, {best_hi:.2f}] -> bar {threshold:.2f}"
                if calibrated else "valley at floor; unchanged"),
    )
    _log.info("Coverage self-calibration: n=%d, %s", n, cal.reason)
    return cal


def apply_calibrated_gate(results, get_recommendation, set_review, get_coverage,
                          floor: float = 0.50, ceil: float = 0.95) -> int:
    """Demote ACCEPT plots whose coverage is below the span-calibrated bar.

    Generic over the result type via accessors so it is trivially unit-testable
    without the heavy pipeline objects. Returns the number of plots demoted. Only
    moves ACCEPT -> REVIEW; never the reverse (0-FP invariant). No-op when
    calibration returns the floor.

    Calibration reads the FULL candidate distribution (so the low coincidental
    cluster is visible and the valley above it is found correctly), but only
    ACCEPT plots below the calibrated bar are demoted.
    """
    if len(results) < MIN_SAMPLES:
        return 0
    cal = calibrate_coverage_threshold([get_coverage(r) for r in results],
                                       floor=floor, ceil=ceil)
    if not cal.calibrated:
        return 0
    demoted = 0
    for r in results:
        if get_recommendation(r) == "ACCEPT" and get_coverage(r) < cal.threshold:
            set_review(r, cal.threshold)
            demoted += 1
    if demoted:
        _log.info("Self-calibration demoted %d ACCEPT->REVIEW (bar raised %.2f->%.2f)",
                  demoted, floor, cal.threshold)
    return demoted
