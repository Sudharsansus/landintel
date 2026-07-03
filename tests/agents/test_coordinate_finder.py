"""CoordinateFinderAgent -- village anchor from web geocode refined by the TNGIS cadastre.

Mocks the two externals (geocode + the vector cadastre) so the test is offline and
deterministic, and asserts the AGENT LOGIC:
  * the survey-number COVERAGE fingerprint decides, the geocode pin only proposes
  * a homonym village (wrong same-named pin) with weak coverage can NOT earn "high"
  * geocode-only is the medium fallback; nothing-found is reported honestly
General -- no village data.
"""
from __future__ import annotations

import landintel.pipeline.m5_cadastral.geo_locate as geo
import landintel.pipeline.m5_cadastral.vector_locate as vloc
from landintel.agents.coordinate_finder import CoordinateFinderAgent


class _Src:
    def __init__(self, polys):
        self._p = polys


def test_decisive_coverage_is_high_confidence(monkeypatch):
    # One pin; the cadastre village carries 11 of 11 FMB surveys -> decisive -> high.
    monkeypatch.setattr(geo, "geocode_candidates", lambda q, limit=5: [(11.30, 77.60)])
    monkeypatch.setattr(vloc, "load_area_parcels_cached", lambda *a, **k: [{"sn": "1", "vc": "V"}])

    def _cands(parcels, surveys, anchor_utm, **kw):
        return [{"vc": "933291", "center": (787866, 1254375), "n_overlap": len(surveys),
                 "dist_m": 500.0, "source": _Src({})}]
    monkeypatch.setattr(vloc, "village_candidates", _cands)

    surveys = {str(n) for n in range(110, 121)}          # 11 surveys
    r = CoordinateFinderAgent().find("NASIYANUR", surveys, district="Erode", taluk="Erode")
    assert r["confidence"] == "high" and r["method"] == "tngis-cadastre"
    assert r["vc"] == "933291" and r["coverage"] == 1.0
    assert 11.2 < r["lat"] < 11.5 and 77.5 < r["lon"] < 77.8


def test_homonym_village_weak_coverage_cannot_be_high(monkeypatch):
    # MOOLAKARAI-like: the first pin is a same-named village elsewhere whose block
    # carries only 3 of 20 FMB surveys (small numbers recur everywhere). Coverage
    # 0.15 < 0.5 -> NOT decisive -> at most "medium", never a false "high".
    monkeypatch.setattr(geo, "geocode_candidates", lambda q, limit=5: [(11.3415, 77.7428)])
    monkeypatch.setattr(vloc, "load_area_parcels_cached", lambda *a, **k: [{"sn": "2", "vc": "W"}])

    def _cands(parcels, surveys, anchor_utm, **kw):
        return [{"vc": "WRONG", "center": (799388, 1255145), "n_overlap": 3,
                 "dist_m": 800.0, "source": _Src({})}]
    monkeypatch.setattr(vloc, "village_candidates", _cands)

    surveys = {str(n) for n in range(1, 21)}             # 20 surveys
    r = CoordinateFinderAgent().find("MOOLAKARAI", surveys, district="Erode", taluk="Erode")
    assert r["confidence"] == "medium"                   # the 07-03 failure, now honest
    assert r.get("coverage", 0) < 0.5


def test_fingerprint_beats_first_pin_across_candidates(monkeypatch):
    # Two geocode candidates (homonym first). The SECOND pin's village carries 15/20
    # surveys vs 3/20 near the first -> the fingerprint picks pin 2's village, high.
    monkeypatch.setattr(geo, "geocode_candidates",
                        lambda q, limit=5: [(11.3415, 77.7428), (11.29, 77.59)]
                        if "MOOLAKARAI" in q else [])
    monkeypatch.setattr(vloc, "load_area_parcels_cached", lambda *a, **k: [{"sn": "2", "vc": "V"}])

    def _cands(parcels, surveys, anchor_utm, **kw):
        if abs(anchor_utm[0] - 799388) < 8000:           # near the wrong (east) pin
            return [{"vc": "WRONG", "center": (799388, 1255145), "n_overlap": 3,
                     "dist_m": 800.0, "source": _Src({})}]
        return [{"vc": "RIGHT", "center": (782500, 1249300), "n_overlap": 15,
                 "dist_m": 600.0, "source": _Src({})}]
    monkeypatch.setattr(vloc, "village_candidates", _cands)

    surveys = {str(n) for n in range(1, 21)}
    r = CoordinateFinderAgent().find("MOOLAKARAI", surveys, district="Erode", taluk="Erode")
    assert r["confidence"] == "high" and r["vc"] == "RIGHT"
    assert r["coverage"] == 0.75                         # 15/20, decisive vs 0.15


def test_geocode_only_is_medium_fallback(monkeypatch):
    monkeypatch.setattr(geo, "geocode_candidates",
                        lambda q, limit=5: [(11.2812, 77.5896)])
    monkeypatch.setattr(vloc, "load_area_parcels_cached", lambda *a, **k: [])
    monkeypatch.setattr(vloc, "village_candidates", lambda *a, **k: [])

    r = CoordinateFinderAgent().find("KANDAMPALAYAM", {"9", "10"}, district="Erode",
                                     taluk="Perundurai")
    assert r["confidence"] == "medium" and r["method"].startswith("geocode:")
    assert r["lat"] == 11.2812 and r["lon"] == 77.5896


def test_nothing_found_is_reported(monkeypatch):
    monkeypatch.setattr(geo, "geocode_candidates", lambda q, limit=5: [])
    r = CoordinateFinderAgent().find("NOWHERE", {"1"}, district="", taluk="")
    assert r["confidence"] == "none" and r["lat"] is None


def test_cadastre_failure_falls_back_to_geocode(monkeypatch):
    monkeypatch.setattr(geo, "geocode_candidates", lambda q, limit=5: [(11.34, 77.64)])

    def _boom(*a, **k):
        raise FileNotFoundError("no parquet")
    monkeypatch.setattr(vloc, "load_area_parcels_cached", _boom)

    r = CoordinateFinderAgent().find("X", {"1"}, district="Erode", taluk="Erode")
    assert r["confidence"] == "medium" and r["lat"] == 11.34
