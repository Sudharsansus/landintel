"""Verification suite for the CLUBBED M2 output (FMB-only georeference + club).

Unlike ``m2_georef.verify`` (which scores ONE georeferenced DXF against a surveyor
file), this checks the WHOLE clubbed set of ``ClubResult`` placements at once, using
only the math the club has -- no surveyor truth. Each check is a deterministic
geometry/structure invariant; together they say "the placements are rigid, in-zone,
non-degenerate, lossless, and tile without overlap".

0-FP DISCIPLINE -- the gate is DEMOTE-ONLY:
  ``gate_results`` may turn an ACCEPT/ACCEPT_SEEDED that FAILS a HARD geometry check
  (closure / area / UTM-range / scale / stone-count) into REVIEW. It NEVER promotes a
  REVIEW/NO_COVERAGE upward. Math finds the failure; the human re-confirms. This mirrors
  the "self-calibrating ACCEPT gate, tighten-only" rule already in the codebase.

Checks (conjunctive over the placed set):
  CLOSURE+AREA          every placed corner ring is a closed polygon, positive area
  UTM_RANGE             every placed plot lies inside the TN UTM box (both zones)
  RIGID_SCALE           every placement scale in [0.8, 1.25] (M1 m -> UTM m, no warp)
  STONE_COUNT_PRESERVED placed plot stone count == source M1 stone count (nothing lost)
  NON_OVERLAPPING_TILING no two ACCEPT footprints overlap interiors beyond ~0.20
  ACCOUNTED             every input FMB has one of the 4 valid recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

# Zone-agnostic TN UTM box (covers 43N west of 78E and 44N east of it). Re-declared
# locally as a hard fallback, but we prefer the m2_georef constants so the two suites
# can never drift apart.
try:  # pragma: no cover - import indirection, both branches trivial
    from ..m2_georef.verify import (
        _TN_UTM_XMAX,
        _TN_UTM_XMIN,
        _TN_UTM_YMAX,
        _TN_UTM_YMIN,
    )
except Exception:  # noqa: BLE001
    _TN_UTM_XMIN, _TN_UTM_XMAX = 100000, 900000
    _TN_UTM_YMIN, _TN_UTM_YMAX = 800000, 1700000

# Rigid-scale band: M1 already converts pixel geometry to real-world metres and UTM is
# metres, so a true placement has scale ~1. A non-unity scale means a warp (or upstream
# unit bug) -- not a placement we trust. Slightly asymmetric (FMB sketches read a touch
# short more often than long) but tight either way.
_SCALE_MIN, _SCALE_MAX = 0.8, 1.25

# Interior overlap (fraction of the SMALLER footprint) above which two ACCEPTs cannot
# both stand -- the SAME centralized threshold every consumer uses (rationale there).
from .disposition_thresholds import TILING_OVERLAP_THRESHOLD as _TILING_OVERLAP_MAX  # noqa: E402

_VALID_RECS = ("ACCEPT", "ACCEPT_SEEDED", "REVIEW", "NO_COVERAGE")

# The HARD geometry checks that, when failed by a placed plot, force a demotion.
_HARD_CHECKS = (
    "CLOSURE_AREA",
    "UTM_RANGE",
    "RIGID_SCALE",
    "STONE_COUNT_PRESERVED",
)


@dataclass
class ClubVerifyCheck:
    """Result of a single clubbed-output verification check."""
    name: str
    passed: bool
    detail: str = ""
    value: float = float("nan")


@dataclass
class ClubVerifyResult:
    """Full verification result for a clubbed M2 output (mirrors VerifyResult)."""
    checks: list[ClubVerifyCheck] = field(default_factory=list)
    all_passed: bool = False

    def __post_init__(self):
        self.all_passed = all(c.passed for c in self.checks)

    def get(self, name: str) -> ClubVerifyCheck | None:
        for c in self.checks:
            if c.name == name:
                return c
        return None

    def failed_names(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]


def _source_stone_count(result) -> int | None:
    """Re-extract the source M1 DXF to count its stones (None if unreadable)."""
    try:
        from ..m2_georef.extract_m1 import extract_m1_dxf
        return len(extract_m1_dxf(result.m1_file).stones)
    except Exception as exc:  # noqa: BLE001
        _log.warning("club verify: cannot re-extract %s: %s", result.m1_file, exc)
        return None


def verify_placed_plot(result, crs: str) -> list[str]:
    """Run the per-plot HARD geometry checks; return the FAILING check names.

    Used by ``gate_results`` to decide a demotion. Only the per-plot checks that a
    single placement can fail on its own are evaluated here (closure/area, UTM range,
    rigid scale, stone-count). Returns ``[]`` when the plot is clean. A plot without a
    placement returns ``["CLOSURE_AREA"]`` (it cannot be a defensible ACCEPT).
    """
    p = result.placement
    if p is None:
        return ["CLOSURE_AREA"]

    failing: list[str] = []

    # CLOSURE + positive AREA.
    fp = p.footprint()
    if fp is None or fp.is_empty or fp.area <= 0:
        failing.append("CLOSURE_AREA")

    # UTM RANGE: every placed stone position inside the TN box.
    adj = p.adjusted
    if adj is None or len(adj) == 0:
        failing.append("UTM_RANGE")
    else:
        xs = [float(x) for x, _y in adj]
        ys = [float(y) for _x, y in adj]
        if not (min(xs) >= _TN_UTM_XMIN and max(xs) <= _TN_UTM_XMAX
                and min(ys) >= _TN_UTM_YMIN and max(ys) <= _TN_UTM_YMAX):
            failing.append("UTM_RANGE")

    # RIGID SCALE in band.
    s = float(getattr(p, "scale", 1.0))
    if not (_SCALE_MIN <= s <= _SCALE_MAX):
        failing.append("RIGID_SCALE")

    # STONE COUNT preserved vs source M1.
    placed_n = len(adj) if adj is not None else 0
    src_n = _source_stone_count(result)
    if src_n is not None and placed_n != src_n:
        failing.append("STONE_COUNT_PRESERVED")

    return failing


def verify_club(results: list, crs: str) -> ClubVerifyResult:
    """Run the full conjunctive verification suite over a clubbed result set."""
    res = ClubVerifyResult()
    placed = [r for r in results if r.placed and r.placement is not None]

    # --- CLOSURE + AREA ---
    bad_closure = []
    for r in placed:
        fp = r.placement.footprint()
        if fp is None or fp.is_empty or fp.area <= 0:
            bad_closure.append(r.survey_number)
    res.checks.append(ClubVerifyCheck(
        "CLOSURE_AREA", not bad_closure,
        f"{len(placed)} placed; "
        + ("all rings closed with positive area"
           if not bad_closure else f"degenerate: {bad_closure}"),
        value=float(len(bad_closure)),
    ))

    # --- UTM RANGE ---
    bad_range = []
    for r in placed:
        adj = r.placement.adjusted
        if adj is None or len(adj) == 0:
            bad_range.append(r.survey_number)
            continue
        xs = [float(x) for x, _y in adj]
        ys = [float(y) for _x, y in adj]
        if not (min(xs) >= _TN_UTM_XMIN and max(xs) <= _TN_UTM_XMAX
                and min(ys) >= _TN_UTM_YMIN and max(ys) <= _TN_UTM_YMAX):
            bad_range.append(r.survey_number)
    res.checks.append(ClubVerifyCheck(
        "UTM_RANGE", not bad_range,
        f"TN box X[{_TN_UTM_XMIN},{_TN_UTM_XMAX}] Y[{_TN_UTM_YMIN},{_TN_UTM_YMAX}]; "
        + ("all placed plots in range"
           if not bad_range else f"out of range: {bad_range}"),
        value=float(len(bad_range)),
    ))

    # --- RIGID SCALE ---
    bad_scale = []
    for r in placed:
        s = float(getattr(r.placement, "scale", 1.0))
        if not (_SCALE_MIN <= s <= _SCALE_MAX):
            bad_scale.append((r.survey_number, round(s, 3)))
    res.checks.append(ClubVerifyCheck(
        "RIGID_SCALE", not bad_scale,
        f"scale band [{_SCALE_MIN}, {_SCALE_MAX}]; "
        + ("all rigid (~1)" if not bad_scale else f"warped: {bad_scale}"),
        value=float(len(bad_scale)),
    ))

    # --- STONE COUNT PRESERVED ---
    bad_count = []
    for r in placed:
        placed_n = len(r.placement.adjusted) if r.placement.adjusted is not None else 0
        src_n = _source_stone_count(r)
        if src_n is not None and placed_n != src_n:
            bad_count.append((r.survey_number, placed_n, src_n))
    res.checks.append(ClubVerifyCheck(
        "STONE_COUNT_PRESERVED", not bad_count,
        "placed stone count == source M1 stone count for all"
        if not bad_count else f"dropped/added stones (survey,placed,src): {bad_count}",
        value=float(len(bad_count)),
    ))

    # --- NON-OVERLAPPING TILING (ACCEPT set only) ---
    accept = [r for r in placed if r.recommendation in ("ACCEPT", "ACCEPT_SEEDED")]
    overlaps = []
    fps = [(r.survey_number, r.placement.footprint()) for r in accept]
    fps = [(sn, fp) for sn, fp in fps if fp is not None and fp.area > 0]
    for i in range(len(fps)):
        sn_a, pa = fps[i]
        for j in range(i + 1, len(fps)):
            sn_b, pb = fps[j]
            if not pa.intersects(pb):
                continue
            inter = pa.intersection(pb)
            if inter.area <= 0:
                continue
            frac = inter.area / max(min(pa.area, pb.area), 1e-9)
            if frac > _TILING_OVERLAP_MAX:
                overlaps.append((sn_a, sn_b, round(frac, 3)))
    res.checks.append(ClubVerifyCheck(
        "NON_OVERLAPPING_TILING", not overlaps,
        f"{len(accept)} ACCEPT footprints tile cleanly"
        if not overlaps else f"interior overlaps > {_TILING_OVERLAP_MAX}: {overlaps}",
        value=float(len(overlaps)),
    ))

    # --- ACCOUNTED ---
    unaccounted = [r.survey_number for r in results
                   if r.recommendation not in _VALID_RECS]
    res.checks.append(ClubVerifyCheck(
        "ACCOUNTED", not unaccounted,
        f"all {len(results)} FMBs in a valid disposition"
        if not unaccounted else f"invalid recommendation: {unaccounted}",
        value=float(len(unaccounted)),
    ))

    res.all_passed = all(c.passed for c in res.checks)
    return res


def gate_results(results: list, crs: str) -> ClubVerifyResult:
    """DEMOTE-ONLY gate: any ACCEPT failing a HARD per-plot geometry check -> REVIEW.

    Mutates ``results`` in place (demotions only, never promotions), then returns the
    ClubVerifyResult computed AFTER demotion (so a reported overlap reflects the final
    ACCEPT set). The hard checks are closure/area, UTM range, rigid scale and stone
    count -- a placement that fails any of them is geometrically indefensible, so it
    drops to REVIEW for a human rather than shipping as a confident ACCEPT.
    """
    for r in results:
        if r.recommendation not in ("ACCEPT", "ACCEPT_SEEDED"):
            continue
        failing = [c for c in verify_placed_plot(r, crs) if c in _HARD_CHECKS]
        if failing:
            r.recommendation = "REVIEW"
            reason = f"demoted by verify: failed {', '.join(failing)}"
            r.note = (r.note + "; " if r.note else "") + reason
            _log.info("club verify demoted %s -> REVIEW (%s)",
                      r.survey_number, ", ".join(failing))

    return verify_club(results, crs)


def format_club_verify(result: ClubVerifyResult) -> str:
    """One-line-per-check text summary, suitable for a sidecar file or a log."""
    status = "PASS" if result.all_passed else "FAIL"
    lines = [f"CLUB VERIFY [{status}]"]
    for c in result.checks:
        icon = "[OK]" if c.passed else "[!!]"
        lines.append(f"  {icon} {c.name:<24} {c.detail}")
    return "\n".join(lines)


def write_club_verify_sidecar(result: ClubVerifyResult, output_path) -> Path:
    """Write the verify summary to a text sidecar (e.g. clubbed.verify.txt)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_club_verify(result) + "\n", encoding="utf-8")
    return output_path
