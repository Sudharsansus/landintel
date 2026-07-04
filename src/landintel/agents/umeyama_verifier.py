"""UmeyamaVerifierAgent -- verifies the MATH of every placement transform + its geometry.

Umeyama / rigid_procrustes must yield a PROPER rigid similarity: an orthonormal rotation with
det = +1 (det = -1 is a REFLECTION -- a mirror-flipped plot, a classic false positive), a
diagnostic scale inside the validated band, and the placed ring must be a valid, SIMPLE
(non-self-intersecting) polygon. ``rigid_procrustes`` enforces det = +1 by construction, so this
is the INVARIANT GUARD that keeps that true forever -- and it also catches a degenerate / bow-tie
ring that the transform check alone cannot.

Error-spotting only (the agent-layer hard rule): it may DEMOTE a placement whose math is unsound
(ACCEPT* -> REVIEW), never promote. No per-village constants -- pure linear-algebra invariants.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .base import Agent, AgentReport, Check, Severity

_ORTHO_TOL = 1e-3          # ||R Rᵀ - I||∞ tolerance for "orthonormal"
_SCALE_LO, _SCALE_HI = 0.5, 2.0
_CONFIDENT = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_RELATIVE")


def _sid(p) -> str:
    return f"{p.village}:{p.survey_number}" if getattr(p, "village", "") else str(p.survey_number)


class UmeyamaVerifierAgent(Agent):
    name = "umeyama_verifier"

    def verify(self, placements) -> AgentReport:
        """Verify + (demote-only) fix each placed plot's transform/ring. Returns an AgentReport."""
        rep = AgentReport(agent=self.name)
        bad_refl, bad_ortho, bad_scale, bad_ring = [], [], [], []

        for p in placements:
            R = getattr(p, "R", None)
            ring_utm = getattr(p, "ring_utm", None)
            if R is None or ring_utm is None:
                continue                                    # not placed -> nothing to verify
            sid = _sid(p)
            R = np.asarray(R, float)
            unsound = False

            if R.shape == (2, 2):
                if float(np.abs(R @ R.T - np.eye(2)).max()) > _ORTHO_TOL:
                    bad_ortho.append(sid); unsound = True
                elif float(np.linalg.det(R)) < 0.0:         # reflection = mirror flip (FP class)
                    bad_refl.append(sid); unsound = True

            s = float(getattr(p, "s_fitted", 1.0))
            if s == s and not (_SCALE_LO < s < _SCALE_HI):
                bad_scale.append(sid)                       # WARN only (an upstream M1 unit bug)

            ring = np.asarray(ring_utm, float)
            poly = Polygon([(float(x), float(y)) for x, y in ring]) if len(ring) >= 3 else None
            if poly is None or (not poly.is_valid) or poly.is_empty or poly.area <= 0:
                bad_ring.append(sid); unsound = True

            if unsound and getattr(p, "disposition", "") in _CONFIDENT:
                p.disposition = "REVIEW"
                p.note = (p.note + " | " if p.note else "") \
                    + "umeyama-verify: unsound transform/ring -> REVIEW"

        rep.checks.append(Check(
            "umeyama_no_reflection", Severity.OK if not bad_refl else Severity.FAIL,
            "no reflected (mirror-flipped) placement" if not bad_refl
            else f"REFLECTION (det(R)<0) -> demoted: {bad_refl}"))
        rep.checks.append(Check(
            "umeyama_orthonormal", Severity.OK if not bad_ortho else Severity.FAIL,
            "all rotations orthonormal" if not bad_ortho
            else f"non-orthonormal R -> demoted: {bad_ortho}"))
        rep.checks.append(Check(
            "umeyama_scale_band", Severity.OK if not bad_scale else Severity.WARN,
            "all diagnostic scales within the validated band" if not bad_scale
            else f"diagnostic scale out of band (possible upstream M1 unit bug): {bad_scale}"))
        rep.checks.append(Check(
            "placed_ring_valid", Severity.OK if not bad_ring else Severity.FAIL,
            "all placed rings are valid simple polygons" if not bad_ring
            else f"degenerate/self-intersecting ring -> demoted: {bad_ring}"))
        n_ok = sum(1 for p in placements if getattr(p, "disposition", "") in _CONFIDENT)
        rep.notes.append(f"transform+ring verified on {n_ok} confident placement(s); "
                         f"rigid_procrustes guarantees det=+1 -> this is the standing invariant guard")
        return rep
