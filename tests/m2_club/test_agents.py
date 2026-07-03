"""M2 agent USE CASES -- general synthetic geometry, NO village data.

Each agent owns one job; these lock its behaviour on synthetic polygons so it is provably
general (not tuned to any village). The agents only measure/sequence/demote -- never promote
-- so 0-FP is structural and tested here too.
"""
from __future__ import annotations

from shapely.geometry import Polygon

from landintel.agents.club_agents import (
    AssemblyAgent,
    ParcelAgent,
    TngisOverlayAgent,
    overlay_gate,
)


def _sq(cx, cy, s):
    h = s / 2.0
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


class _Cad:
    """Minimal cadastral source: survey -> parcel polygon."""
    def __init__(self, parcels):
        self._p = parcels

    def get(self, sv, village=None):
        class P:
            polygon = self._p.get(sv)
        return P() if sv in self._p else None


# --------------------------------------------------------------------------- ParcelAgent
def test_parcel_agent_flags_sliver_and_merged():
    parcels = {sv: _sq(0, 0, 100) for sv in ("1", "2", "3", "4")}   # ~10000 m2 each
    parcels["5"] = _sq(500, 0, 10)                                  # sliver ~100 m2
    parcels["6"] = _sq(0, 500, 300)                                 # merged ~90000 m2
    r = ParcelAgent().run(_Cad(parcels), {"1", "2", "3", "4", "5", "6"})
    assert r.ok
    assert r.data["flagged"] == {"5", "6"}                          # sliver + merged flagged
    assert r.confidence < 1.0


def test_parcel_agent_none_when_empty():
    r = ParcelAgent().run(_Cad({}), {"1", "2"})
    assert not r.ok and "no cadastral parcels" in r.issues[0]


# ----------------------------------------------------------------------- TngisOverlayAgent
def test_overlay_perfect_match_iou_one():
    par = {"1": _sq(0, 0, 100), "2": _sq(200, 0, 100)}
    fp = {"1": _sq(0, 0, 100), "2": _sq(200, 0, 100)}              # exact overlay
    r = TngisOverlayAgent().run(fp, par)
    assert r.data["mean_iou"] > 0.99
    assert r.data["overlap_frac"] < 1e-6                            # non-overlapping tiling


def test_overlay_offset_lowers_iou():
    par = {"1": _sq(0, 0, 100)}
    fp = {"1": _sq(50, 0, 100)}                                     # shifted half a width
    r = TngisOverlayAgent().run(fp, par)
    assert 0.2 < r.data["mean_iou"] < 0.5                           # partial overlap


def test_overlay_detects_stacked_plots():
    par = {"1": _sq(0, 0, 100), "2": _sq(10, 0, 100)}
    fp = {"1": _sq(0, 0, 100), "2": _sq(10, 0, 100)}               # two plots nearly on top
    r = TngisOverlayAgent().run(fp, par)
    assert r.data["overlap_frac"] > 0.1
    assert any("overlap" in s for s in r.issues)


# --------------------------------------------------------------------------- AssemblyAgent
def test_assembly_demotes_lower_confidence_of_overlapping_pair():
    fp = {"1": _sq(0, 0, 100), "2": _sq(20, 0, 100), "3": _sq(1000, 0, 100)}
    r = AssemblyAgent().run(fp, accepted={"1", "2", "3"},
                            confidence={"1": 0.9, "2": 0.5, "3": 0.8})
    assert r.data["demote"] == {"2"}                                # lower-conf of 1&2
    assert r.data["accept"] == {"1", "3"}


def test_assembly_keeps_a_clean_tiling():
    fp = {"1": _sq(0, 0, 100), "2": _sq(200, 0, 100)}             # share no interior
    r = AssemblyAgent().run(fp, accepted={"1", "2"}, confidence={"1": 0.9, "2": 0.9})
    assert r.data["demote"] == set()
    assert r.data["accept"] == {"1", "2"}


# --------------------------------------------------------------------------- overlay_gate (0-FP)
def _gate(rec, iou, **kw):
    kw.setdefault("seated", True)
    kw.setdefault("has_placement", True)
    kw.setdefault("in_demote", False)
    kw.setdefault("is_vector_parcel", True)
    kw.setdefault("contested", False)
    return overlay_gate(rec, iou, **kw)


def test_gate_strong_overlay_promotes_even_when_off_seat():
    # 47-like: high IoU on its OWN uncontested vector parcel, but flagged off-seat -> ACCEPT.
    rec, reason = _gate("REVIEW", 0.73, seated=False)
    assert rec == "ACCEPT" and "strong" in reason


def test_gate_seated_path_unchanged():
    rec, _ = _gate("REVIEW", 0.55, seated=True)          # original path still promotes
    assert rec == "ACCEPT"


def test_gate_strong_needs_uncontested_vector_parcel():
    # contested label (a mislabel collision) blocks the strong path -> stays REVIEW.
    assert _gate("REVIEW", 0.73, seated=False, contested=True)[0] == "REVIEW"
    # non-vector (label-point box) parcel is not strong-eligible either.
    assert _gate("REVIEW", 0.73, seated=False, is_vector_parcel=False)[0] == "REVIEW"


def test_gate_never_promotes_low_iou_or_off_seat_only():
    assert _gate("REVIEW", 0.30, seated=True)[0] == "REVIEW"           # too low
    assert _gate("REVIEW", 0.55, seated=False)[0] == "REVIEW"          # seated path needs seated
    assert _gate("REVIEW", 0.90, in_demote=True)[0] == "REVIEW"        # overlap-demoted stays
    assert _gate("REVIEW", 0.90, has_placement=False)[0] == "REVIEW"   # no placement


def test_gate_demotes_self_contradicting_accept():
    # 141-like: ACCEPT but ~0 overlap with its OWN real vector parcel -> REVIEW.
    rec, reason = _gate("ACCEPT", 0.00)
    assert rec == "REVIEW" and "contradiction" in reason


def test_gate_keeps_good_accept_and_ignores_boxed_parcel():
    assert _gate("ACCEPT", 0.80)[0] == "ACCEPT"                        # good ACCEPT kept
    # a boxed (non-vector) parcel with low IoU is NOT demoted (no trustworthy geometry).
    assert _gate("ACCEPT", 0.00, is_vector_parcel=False)[0] == "ACCEPT"


def test_gate_same_parcel_containment_promotes_subdivision():
    # 14-like: cadastre parcel 99% inside the FMB footprint, extent diverged 2.3x, IoU 0.43 (<0.5)
    # -> same land (parent/subdivision), decisive lock -> ACCEPT.
    rec, reason = _gate("REVIEW", 0.43, seated=False, containment=0.99,
                        area_factor=2.30, decisive_lock=True)
    assert rec == "ACCEPT" and "same parcel" in reason


def test_gate_same_parcel_needs_decisive_lock():
    # identical geometry but the village lock is NOT decisive -> stays REVIEW (could be wrong village)
    assert _gate("REVIEW", 0.43, seated=False, containment=0.99, area_factor=2.30,
                 decisive_lock=False)[0] == "REVIEW"


def test_gate_same_parcel_rejects_gross_size_mismatch():
    # 55-like: cadastre tiny fragment fully inside a 10x-bigger FMB -> factor 10 > 2.5 -> REVIEW
    # (a renumbered/different parcel, not a subdivision remnant).
    assert _gate("REVIEW", 0.10, seated=False, containment=1.0, area_factor=10.5,
                 decisive_lock=True)[0] == "REVIEW"
    # 37-like: only partially contained (0.64 < 0.90) -> REVIEW even with decisive lock.
    assert _gate("REVIEW", 0.34, seated=False, containment=0.64, area_factor=1.51,
                 decisive_lock=True)[0] == "REVIEW"
