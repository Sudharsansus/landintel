"""CompositeCadastralSource: primary preferred, secondary fallback, candidates unioned, 0-FP."""
from __future__ import annotations

from shapely.geometry import Polygon

from landintel.pipeline.m5_cadastral.composite import CompositeCadastralSource
from landintel.pipeline.m5_cadastral.source import CadastralParcel


def _parcel(sn, area_side):
    return CadastralParcel(survey_number=sn,
                           polygon=Polygon([(0, 0), (area_side, 0),
                                            (area_side, area_side), (0, area_side)]),
                           source_crs="EPSG:32643")


class _Src:
    def __init__(self, parcels=None, cands=None, labels=None, aggr=None, crs="EPSG:32643"):
        self._p = parcels or {}
        self._c = cands or {}
        self._l = labels or {}
        self._a = set(aggr or [])
        self.crs = crs

    def get(self, sn, village=None):
        return self._p.get(sn)

    def recovered_candidates(self, sn):
        return self._c.get(sn, [])

    def label_point(self, sn):
        return self._l.get(sn)

    def is_aggressive(self, sn):
        return sn in self._a

    def survey_numbers(self):
        return set(self._p)


def test_primary_preferred_secondary_fallback():
    primary = _Src(parcels={"10": _parcel("10", 50)}, labels={"10": (1.0, 2.0)})
    secondary = _Src(parcels={"10": _parcel("10", 99), "20": _parcel("20", 30)},
                     labels={"10": (9.0, 9.0), "20": (5.0, 6.0)})
    c = CompositeCadastralSource(primary, secondary)

    # primary owns "10" -> its parcel + label win
    assert c.get("10").polygon.area == 50 * 50
    assert c.label_point("10") == (1.0, 2.0)
    # secondary-only "20" falls back
    assert c.get("20").polygon.area == 30 * 30
    assert c.label_point("20") == (5.0, 6.0)
    # union of survey numbers
    assert c.survey_numbers() == {"10", "20"}


def test_recovered_candidates_union_without_duplicating_secondary_get():
    primary = _Src(parcels={"10": _parcel("10", 50)},
                   cands={"10": [_parcel("10", 40)]})
    secondary = _Src(parcels={"10": _parcel("10", 99)},
                     cands={"10": [_parcel("10", 60)]})
    c = CompositeCadastralSource(primary, secondary)
    areas = sorted(p.polygon.area for p in c.recovered_candidates("10"))
    # primary cand (40^2) + secondary parcel (99^2) + secondary cand (60^2)
    assert areas == sorted([40 * 40, 99 * 99, 60 * 60])

    # when primary has NO parcel, secondary's parcel comes via get(), NOT duplicated in candidates
    primary2 = _Src(parcels={})
    c2 = CompositeCadastralSource(primary2, secondary)
    cand_areas = [p.polygon.area for p in c2.recovered_candidates("10")]
    assert 99 * 99 not in cand_areas and cand_areas == [60 * 60]


def test_is_aggressive_only_when_no_clean_alternative():
    # "10": primary aggressive BUT secondary clean -> NOT aggressive (a clean parent exists, so
    # the gate must not demote it just because the other source's sub-cell was aggressive -- this
    # is the survey-698 fix: TNGIS aggressive sub-cell + S3 clean full parent => ACCEPT-able).
    # "20": only an aggressive secondary -> aggressive. "30": aggressive primary, no secondary
    # parcel -> aggressive. "99": unknown -> not aggressive.
    primary = _Src(parcels={"10": _parcel("10", 50), "30": _parcel("30", 40)}, aggr={"10", "30"})
    secondary = _Src(parcels={"10": _parcel("10", 99), "20": _parcel("20", 30)}, aggr={"20"})
    c = CompositeCadastralSource(primary, secondary)
    assert c.is_aggressive("10") is False           # clean secondary parent exists
    assert c.is_aggressive("20") is True            # only an aggressive parcel anywhere
    assert c.is_aggressive("30") is True            # only an aggressive primary, no clean alt
    assert c.is_aggressive("99") is False           # unknown -> not aggressive
