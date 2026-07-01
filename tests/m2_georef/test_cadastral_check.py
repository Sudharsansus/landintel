"""Cadastral cross-validation -- the independent ground-truth accuracy/FP check.

The math gates decide placement; this checks each ACCEPT plot against its authoritative cadastral
parcel. 0-FP invariant under test: it CONFIRMS good placements and DEMOTES gross misses
(ACCEPT->REVIEW), but NEVER promotes anything.
"""
from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon

from landintel.pipeline.m2_georef.cadastral_check import (apply_cadastral_gate,
                                                         cadastral_agreement)


def _sq(cx, cy, side):
    h = side / 2
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


# --- agreement scoring ------------------------------------------------------
def test_well_placed_is_confirmed():
    parcel = _sq(700000, 1200000, 50)
    placed = _sq(700001, 1200001, 48)            # almost coincident
    ag = cadastral_agreement(placed, parcel, "10")
    assert ag.status == "confirmed" and ag.centroid_inside and ag.iou > 0.7


def test_subdivision_inside_parcel_is_confirmed():
    parcel = _sq(700000, 1200000, 50)
    placed = _sq(700010, 1200010, 15)            # small subdivision, inside the parcel
    ag = cadastral_agreement(placed, parcel, "10")
    assert ag.status == "confirmed"              # centroid inside / contained
    assert ag.contained_frac > 0.9


def test_far_placement_is_gross_miss():
    parcel = _sq(700000, 1200000, 50)
    placed = _sq(700500, 1200500, 50)            # 700m away, no overlap
    ag = cadastral_agreement(placed, parcel, "10")
    assert ag.status == "gross_miss"
    assert ag.iou == 0.0 and not ag.centroid_inside


def test_partial_overlap_is_weak_not_demoted():
    parcel = _sq(700000, 1200000, 50)
    placed = _sq(700040, 1200000, 50)            # genuinely overlaps (~20%), centroid just outside
    ag = cadastral_agreement(placed, parcel, "10")
    assert 0 < ag.iou < 0.15 and not ag.centroid_inside
    assert ag.status == "weak"                   # near + overlapping -> not demoted, not confirmed


def test_no_parcel_is_safe():
    assert cadastral_agreement(None, _sq(0, 0, 10), "10").status == "no_parcel"
    assert cadastral_agreement(_sq(0, 0, 10), None, "10").status == "no_parcel"


# --- the gate: demote-only, never promote -----------------------------------
@dataclass
class R:
    survey_number: str
    recommendation: str
    output_file: str = ""
    error: str = ""


class Cad:
    def __init__(self, parcels): self._p = parcels
    def get(self, sn, village=None):
        return self._p.get(sn)


class _Parcel:
    def __init__(self, poly): self.polygon = poly


def test_gate_confirms_and_demotes_but_never_promotes():
    parcels = {
        "good": _Parcel(_sq(700000, 1200000, 50)),
        "bad":  _Parcel(_sq(701000, 1200000, 50)),   # far from where 'bad' is placed
    }
    footprints = {
        "good": _sq(700000, 1200000, 48),            # on its parcel -> confirm
        "bad":  _sq(700000, 1200000, 48),            # placed 1km from its parcel -> gross miss
    }
    results = [R("good", "ACCEPT"), R("bad", "ACCEPT"), R("rev", "REVIEW")]

    demoted = []
    stats = apply_cadastral_gate(
        results, Cad(parcels),
        footprint_fn=lambda r: footprints.get(r.survey_number),
        on_demote=lambda r, ag: (setattr(r, "recommendation", "REVIEW"), demoted.append(r.survey_number)),
        on_confirm=lambda r, ag: None)

    assert stats["confirmed"] == 1 and stats["gross_miss_demoted"] == 1
    assert results[0].recommendation == "ACCEPT"      # good stays ACCEPT
    assert results[1].recommendation == "REVIEW"      # bad demoted
    assert results[2].recommendation == "REVIEW"      # untouched (was REVIEW)
    assert demoted == ["bad"]
    # never promotes: no result moved up to ACCEPT
    assert all(r.recommendation in ("ACCEPT", "REVIEW") for r in results)


def test_gate_noop_without_cadastral():
    results = [R("1", "ACCEPT")]
    assert apply_cadastral_gate(results, None, footprint_fn=lambda r: None)["checked"] == 0
    assert results[0].recommendation == "ACCEPT"
