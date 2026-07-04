"""M3 operator loop-closer -- turns the InputRequest worklist answers into CLOSED plots.

This is the last mile of the agent layer's "100% is a process" design: the product emits,
per unplaced plot, the ONE minimal input that closes it (a human confirm for a LOCATED plot, or
2 corner-stone -> UTM points for one with no auto position); the operator provides it; and THIS
turns those answers into confident placements. It is FP-safe because the HUMAN supplies the
identity and the deterministic gates still decide:

  * confirm : only a plot the math already LOCATED (has geometry, cadastre+stone agreement) can be
              confirmed -- a human vouches for a placement the pipeline already produced.
  * seed    : two operator corner->UTM correspondences fully determine a RIGID (scale-locked, rule
              2) placement; a too-short baseline is rejected (amplifies operator point error), so a
              bad seed is forced to REVIEW, never ACCEPT.

Never a geometric guess -- the agent adds NOTHING the human did not supply. Disposition of a
closed plot is ACCEPT_SEEDED (distinct from the auto ACCEPT / ACCEPT_RELATIVE tiers).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from .extract_m1 import extract_m1_dxf
from .m3_deliverables import M3Placement, place_scale_locked

_log = logging.getLogger(__name__)

# A 2-corner seed is exactly determined (no averaging), so a SHORT baseline amplifies the
# operator's point error across the whole plot. Reject below this (matches seed_place's bound).
SEED_MIN_BASELINE_M = 5.0


def _sid(p: M3Placement) -> str:
    return f"{p.village}:{p.survey_number}" if p.village else str(p.survey_number)


def _apply_confirms(placements, confirms) -> list[str]:
    """Promote each operator-CONFIRMED plot REVIEW->ACCEPT_SEEDED. Only a LOCATED plot (already
    has geometry) can be confirmed -- the human vouches for a placement the math produced, never
    invents one. Returns the survey ids closed."""
    want = {str(s) for s in confirms}
    closed = []
    for p in placements:
        if _sid(p) in want or str(p.survey_number) in want:
            if p.disposition in ("REVIEW",) and p.ring_utm is not None and len(p.ring_utm) >= 3:
                p.disposition = "ACCEPT_SEEDED"
                p.note = (p.note + " | " if p.note else "") + "operator-confirmed (human vouches)"
                closed.append(_sid(p))
    return closed


def _seed_one(m1_path, corner_a, corner_b, utm_a, utm_b):
    """RIGID (scale-locked, rule 2) placement of ONE plot from two operator corner->UTM points.
    Returns (ring_utm, n_corners, note) or None (labels missing / baseline too short)."""
    m1 = extract_m1_dxf(str(m1_path))
    label_idx: dict[str, int] = {}
    for st in m1.stones:
        label_idx.setdefault(str(st.label), st.index)
    if str(corner_a) not in label_idx or str(corner_b) not in label_idx:
        return None
    ia, ib = label_idx[str(corner_a)], label_idx[str(corner_b)]
    src = np.array([[m1.stones[ia].x, m1.stones[ia].y],
                    [m1.stones[ib].x, m1.stones[ib].y]], float)
    dst = np.array([list(utm_a), list(utm_b)], float)
    if float(np.hypot(*(dst[0] - dst[1]))) < SEED_MIN_BASELINE_M:
        return None                                  # short baseline -> reject (never ACCEPT)
    R, t, _s, _res = place_scale_locked(src, dst)    # scale locked to 1 (rule 2)
    pos = m1.stone_positions()
    ring = (pos @ R.T + t)[np.array(m1.outer_stone_indices)]
    return ring, len(m1.outer_stone_indices), (
        f"operator 2-corner seed ({corner_a}->{utm_a}, {corner_b}->{utm_b}), rigid s=1")


def _apply_seeds(placements, seeds, m1_by_survey) -> list[str]:
    """Close each seeded plot: place it RIGIDLY from the two operator corners and set
    ACCEPT_SEEDED. Replaces the plot's prior REVIEW/NEEDS_GPS geometry. Returns ids closed."""
    by_sid = {_sid(p): p for p in placements}
    by_sv = {str(p.survey_number): p for p in placements}
    closed = []
    for s in seeds:
        key = str(s.get("survey", ""))
        p = by_sid.get(key) or by_sv.get(key.split(":")[-1])
        if p is None:
            continue
        m1_path = m1_by_survey.get(str(p.survey_number)) or m1_by_survey.get(key.split(":")[-1])
        if not m1_path:
            continue
        try:
            placed = _seed_one(m1_path, s["corner_a"], s["corner_b"],
                               tuple(s["utm_a"]), tuple(s["utm_b"]))
        except Exception as exc:  # noqa: BLE001
            _log.warning("seed for %s failed: %s", key, exc)
            placed = None
        if placed is None:
            continue
        ring, ncorners, note = placed
        p.disposition = "ACCEPT_SEEDED"
        p.ring_utm = ring
        p.n_matched = 2
        p.n_corners = ncorners
        p.note = note
        closed.append(_sid(p))
    return closed


def write_field_worklist(workdir):
    """Turn input_requests.json (the agent layer's path-to-100% worklist) into a FILLABLE field
    CSV the survey team prints and completes. One row per plot needing input, most-impactful first;
    the blank columns map directly back to operator_confirms.json / operator_seeds.json so the
    loop-closer can ingest the answers. Returns the CSV path, or None if there is no worklist."""
    workdir = Path(workdir)
    src = workdir / "input_requests.json"
    if not src.exists():
        return None
    try:
        reqs = json.loads(src.read_text()).get("requests", [])
    except Exception:  # noqa: BLE001
        return None
    out = workdir / "field_worklist.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["priority", "plot(village:survey)", "needs", "reason",
                    "known_x_utm", "known_y_utm", "CONFIRM(yes/no)",
                    "cornerA_label", "cornerA_x_utm", "cornerA_y_utm",
                    "cornerB_label", "cornerB_x_utm", "cornerB_y_utm"])
        for i, r in enumerate(reqs, 1):
            it = r.get("input_type", "")
            known = r.get("known_utm") or [None, None]
            needs = ("CONFIRM (1 human yes/no)" if it == "confirm_placement"
                     else "2 GPS corner points" if it == "two_corner_seed" else it)
            w.writerow([i, r.get("survey_number", ""), needs, (r.get("reason", "") or "")[:90],
                        known[0] if known else "", known[1] if known else "",
                        "", "", "", "", "", "", ""])
    return out


def apply_operator_inputs(placements, m1_by_survey, workdir) -> dict:
    """Close worklisted plots from operator answers found in ``workdir`` (if present):

      operator_confirms.json : ["MOOLAKARAI:17", ...]                     (human-confirmed LOCATED)
      operator_seeds.json    : [{"survey":"KANDAMPALAYAM:52","corner_a":"1","corner_b":"3",
                                 "utm_a":[x,y], "utm_b":[x,y]}, ...]        (2 corner GPS points)

    Mutates ``placements`` in place; returns {"confirmed":[...], "seeded":[...]}. FP-safe: the
    human supplies the identity, the deterministic gates decide, agents add no geometric guess.
    Absent files -> no-op (the automatic result stands)."""
    workdir = Path(workdir)
    out = {"confirmed": [], "seeded": []}
    cf = workdir / "operator_confirms.json"
    sf = workdir / "operator_seeds.json"
    if cf.exists():
        try:
            out["confirmed"] = _apply_confirms(placements, json.loads(cf.read_text()))
        except Exception as exc:  # noqa: BLE001
            _log.error("operator_confirms ingest failed: %s", exc)
    if sf.exists():
        try:
            out["seeded"] = _apply_seeds(placements, json.loads(sf.read_text()), m1_by_survey)
        except Exception as exc:  # noqa: BLE001
            _log.error("operator_seeds ingest failed: %s", exc)
    return out
