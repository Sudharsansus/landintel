"""Propose -> RE-GATE -- the loop that lets the LLM "fine-tune like Claude" safely.

The reasoning agent proposes a bounded automation to re-try (flip the upside-down label,
run road-closure recovery, re-OCR at more angles, corroborate by topology). This module
RE-RUNS that deterministic fix and reads back THE GATE'S verdict -- the LLM never sets a
placement. Two independent safety layers make the loop 0-FP by construction:

  1. The single-plot re-run uses the SAME deterministic gate as the first pass; a fix only
     "sticks" if the gate accepts it on the evidence.
  2. A GLOBAL guard then re-checks the FP invariants (the new confident plot must not overlap
     any other confident footprint and must not be cross-village). If accepting it would
     break the tiling, the plot is REVERTED -- so even a gate-accepted re-run cannot create a
     false positive at the job level.

If no re-attempt hook is wired for this run (e.g. the agent-only path, or no cadastral
source available), AUTO proposals are recorded as DEFERRED -- honestly, not silently
accepted. The fix then surfaces as the operator input request instead.
"""

from __future__ import annotations

import logging

from .concept import AUTO_ACTIONS

_log = logging.getLogger(__name__)
_CONF = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")
from ..pipeline.m2_club.disposition_thresholds import (  # noqa: E402
    TILING_OVERLAP_THRESHOLD as _OVERLAP_FRAC,
)


def _footprint(r):
    """Stage-agnostic footprint: prefer the new-M2 in-memory placement Polygon; else read
    it back from the georef output DXF (M3). Returns a valid, positive-area Polygon or None."""
    p = getattr(r, "placement", None)
    if p is not None and not hasattr(r, "match_method"):   # ClubResult
        try:
            fp = p.footprint()
        except Exception:  # noqa: BLE001
            fp = None
        return fp if (fp is not None and fp.is_valid and fp.area > 0) else None
    from ..pipeline.m2_georef.pipeline import _footprint_polygon
    if not getattr(r, "output_file", ""):
        return None
    fp = _footprint_polygon(r.output_file)
    return fp if (fp is not None and fp.is_valid and fp.area > 0) else None


def _breaks_fp_invariant(r, results, village) -> str | None:
    """Return a reason string if marking ``r`` confident would violate a global FP
    invariant (overlap with another confident footprint, or cross-village); else None."""
    from .dispositions import is_cross_village
    if village and is_cross_village(r, village):
        return "cross-village plot may not be confident"
    fp = _footprint(r)
    if fp is None:
        return None
    for other in results:
        if other is r or other.recommendation not in _CONF:
            continue
        of = _footprint(other)
        if of is None or not fp.intersects(of):
            continue
        frac = fp.intersection(of).area / max(min(fp.area, of.area), 1e-9)
        if frac > _OVERLAP_FRAC:
            return f"would overlap confident plot {other.survey_number} ({frac:.0%})"
    return None


def regate_proposals(results, proposals, context: dict | None = None) -> dict:
    """Re-run each AUTO proposal through the deterministic gate; record the verdict.

    ``context['reattempt']`` (optional) is a callable ``(result, action) -> (bool, str)``
    that re-runs the deterministic fix for ONE plot and returns (gate_accepted, detail).
    The pipeline supplies one bound to the cadastral source (see build_cadastral_reattempt);
    tests supply a stub. Returns a summary; mutates the Proposal objects in place.
    """
    context = context or {}
    village = context.get("village", "INGUR")
    reattempt = context.get("reattempt")
    by_sn = {r.survey_number: r for r in results}

    n_accepted = n_deferred = n_rejected = 0
    for p in proposals:
        if not p.is_auto:
            continue
        r = by_sn.get(p.survey_number)
        if r is None:
            continue
        if reattempt is None:
            p.regated = False
            p.note = (f"deferred: '{p.action}' is a re-runnable automation but no re-attempt "
                      f"hook is wired in this run -> surfaced as an operator input request.")
            n_deferred += 1
            continue

        prior = r.recommendation
        try:
            accepted, detail = reattempt(r, p.action)
        except Exception as exc:  # noqa: BLE001 - a re-run must never crash the job
            p.regated, p.note = True, f"re-run errored ({exc}); left as-is (0-FP preserved)."
            n_rejected += 1
            continue

        p.regated = True
        if not accepted:
            # the deterministic re-run did not improve it; the gate still says no
            if r.recommendation in _CONF:           # defensive: shouldn't happen
                r.recommendation = prior
            p.accepted_by_gate = False
            p.note = f"re-ran '{p.action}': gate still REVIEW ({detail}); escalate to input."
            n_rejected += 1
            continue

        # the single-plot gate accepted -> now the GLOBAL FP guard must also pass
        breach = _breaks_fp_invariant(r, results, village)
        if breach:
            r.recommendation = prior                 # REVERT: 0-FP outranks recall
            p.accepted_by_gate = False
            p.note = (f"gate accepted '{p.action}' but global FP guard blocked it ({breach}) "
                      f"-> reverted to {prior}. 0 false positives preserved.")
            n_rejected += 1
        else:
            p.accepted_by_gate = True
            p.note = (f"'{p.action}' re-gated: GATE accepted ({detail}); FP invariants "
                      f"re-checked OK. Recommendation set by the gate, not the LLM.")
            n_accepted += 1

    return {"accepted": n_accepted, "rejected": n_rejected, "deferred": n_deferred}


def build_cadastral_reattempt(cadastral_source, surveyor, output_dir, crs):
    """Production re-attempt hook: re-run the deterministic cadastral placement for a single
    plot and report whether the GATE made it confident. The proposed AUTO action selects the
    recovery emphasis; the placement code already tries label-rotation / open-parcel recovery
    internally, so this re-runs that path and lets the unchanged gate return the verdict.

    Returns a callable ``(result, action) -> (gate_accepted: bool, detail: str)``.
    """
    from ..pipeline.m2_georef.pipeline import _place_by_cadastral

    def _reattempt(r, action):
        if action not in AUTO_ACTIONS:
            return False, f"'{action}' is not an automation"
        try:
            _place_by_cadastral([r], cadastral_source, surveyor, output_dir, crs)
        except Exception as exc:  # noqa: BLE001
            return False, f"re-run failed: {exc}"
        ok = r.recommendation in _CONF
        return ok, (f"disposition now {r.recommendation}, "
                    f"cad_residual={getattr(r, 'cad_residual', float('inf')):.0f}m")

    return _reattempt
