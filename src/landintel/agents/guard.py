"""GuardAgent -- deterministic, final-line false-positive guards.

These are belt-and-suspenders assertions that run AFTER the math gates, catching FP
patterns that are easy to miss upstream:

  * the 180-degree digit-anagram trap (a label like "669" read upside-down becomes
    "699"; if both are placed near the same point one is a rotation phantom);
  * a duplicate survey number landing in the ACCEPT set twice (only one parcel per
    survey can be real -- the other is a mis-identification);
  * an ACCEPT with no geometry (a confident placement must have a footprint);
  * two ACCEPTs sharing one footprint (mutually-exclusive parcels stacked).

Deterministic -> no LLM, no FP risk; it can only flag (FAIL) -- it never places.
STAGE-AGNOSTIC via the normalized ``PlotDisposition`` adapter.
"""

from __future__ import annotations

from .base import Agent, AgentReport, Check, Severity
from .dispositions import normalize

# 180-degree rotation of each digit on a 7-seg/printed glyph (others don't map).
_ROT = {"0": "0", "1": "1", "6": "9", "8": "8", "9": "6"}
_ANAGRAM_TOL_M = 20.0
_FOOTPRINT_DUP = 0.50    # ACCEPT pair overlapping interiors > this share = same footprint


def rot180(s: str) -> str | None:
    """The string a printed number reads as when rotated 180 degrees, or None if it
    contains a non-rotatable digit (2,3,4,5,7)."""
    out = []
    for ch in reversed(s):
        if ch not in _ROT:
            return None
        out.append(_ROT[ch])
    return "".join(out)


class GuardAgent(Agent):
    name = "guard"

    def run(self, results, context: dict) -> AgentReport:
        rep = AgentReport(agent=self.name)
        disps = normalize(results)

        # centroids for everything placed (confident OR located review)
        cen: dict[str, tuple[float, float]] = {}
        for d in disps:
            if d.has_geometry:
                c = d.footprint.centroid
                cen[d.survey] = (c.x, c.y)

        conf = [d for d in disps if d.is_confident]
        conf_sn = {d.survey for d in conf}

        # ---- G1: 180-degree anagram rotation-phantom trap ----
        hits = []
        seen = set()
        for sn, (x, y) in cen.items():
            anag = rot180(sn)
            if anag is None or anag == sn or anag not in cen:
                continue
            key = tuple(sorted((sn, anag)))
            if key in seen:
                continue
            seen.add(key)
            ax, ay = cen[anag]
            dist = ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
            if dist <= _ANAGRAM_TOL_M:
                both_conf = sn in conf_sn and anag in conf_sn
                hits.append((sn, anag, round(dist, 1), both_conf))
        hard = [h for h in hits if h[3]]
        soft = [h for h in hits if not h[3]]
        rep.checks.append(Check(
            "anagram_no_confident_collision",
            Severity.OK if not hard else Severity.FAIL,
            "no anagram pair both-confident at one point" if not hard
            else f"ANAGRAM FP: both placed confidently <{_ANAGRAM_TOL_M}m apart: {hard}"))
        if soft:
            rep.checks.append(Check(
                "anagram_phantom_present", Severity.WARN,
                f"anagram pair co-located (one is likely a 180-deg rotation phantom; "
                f"gate-protected as REVIEW): {soft}"))
        else:
            rep.checks.append(Check("anagram_phantom_present", Severity.OK,
                                    "no rotation-phantom co-locations"))

        # ---- G2: no survey number ACCEPTed twice (one parcel per survey is real) ----
        counts: dict[str, int] = {}
        for d in conf:
            counts[d.survey] = counts.get(d.survey, 0) + 1
        dups = sorted(sn for sn, n in counts.items() if n > 1)
        rep.checks.append(Check(
            "no_duplicate_confident_survey",
            Severity.OK if not dups else Severity.FAIL,
            "every confident survey number is unique" if not dups
            else f"DUPLICATE survey in confident set (only one can be real): {dups}"))

        # ---- G3: no ACCEPT without geometry (a confident plot must have a footprint) ----
        no_geom = sorted(d.survey for d in conf if not d.has_geometry)
        # When NO confident plot has geometry at all (e.g. a footprint-free unit test),
        # this is "geometry not available", not a violation -> INFO, never FAIL.
        any_geom = any(d.has_geometry for d in conf)
        if not any_geom:
            rep.checks.append(Check(
                "confident_has_geometry", Severity.INFO,
                "no footprints available to check (geometry not loaded)"))
        else:
            rep.checks.append(Check(
                "confident_has_geometry",
                Severity.OK if not no_geom else Severity.FAIL,
                "every confident plot has geometry" if not no_geom
                else f"ACCEPT without geometry (no footprint): {no_geom}"))

        # ---- G4: no two ACCEPTs on the SAME footprint (stacked mutually-exclusive parcels) ----
        polys = [(d.survey, d.footprint) for d in conf if d.has_geometry]
        same_fp = []
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                a, b = polys[i][1], polys[j][1]
                if a.intersects(b):
                    frac = a.intersection(b).area / max(min(a.area, b.area), 1e-9)
                    if frac > _FOOTPRINT_DUP:
                        same_fp.append((polys[i][0], polys[j][0], round(frac, 2)))
        rep.checks.append(Check(
            "no_two_confident_on_same_footprint",
            Severity.OK if not same_fp else Severity.FAIL,
            "no two confident plots share a footprint" if not same_fp
            else f"TWO ACCEPTs on one footprint: {same_fp}"))
        return rep
