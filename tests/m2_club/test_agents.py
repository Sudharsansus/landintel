"""M2 agent USE CASES -- general synthetic geometry, NO village data.

Each agent owns one job; these lock its behaviour on synthetic polygons so it is provably
general (not tuned to any village). The agents only measure/sequence/demote -- never promote
-- so 0-FP is structural and tested here too.
"""
from __future__ import annotations

from shapely.geometry import Polygon

from landintel.pipeline.m2_club.agents import (
    AssemblyAgent,
    ParcelAgent,
    TngisOverlayAgent,
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
