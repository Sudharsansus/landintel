"""CoordinateFinderAgent -- village anchor from web geocode refined by the TNGIS cadastre.

Mocks the two externals (geocode + the vector cadastre) so the test is offline + deterministic
and asserts the AGENT LOGIC: cadastre match wins (exact, high), geocode-only is the medium
fallback, and nothing-found is reported honestly. General -- no village data.
"""
from __future__ import annotations

import landintel.pipeline.m5_cadastral.geo_locate as geo
import landintel.pipeline.m5_cadastral.vector_locate as vloc
from landintel.agents.coordinate_finder import CoordinateFinderAgent


class _Src:
    def __init__(self, polys):
        self._p = polys


def test_cadastre_match_is_high_confidence_and_exact(monkeypatch):
    # rough geocode returns a coarse pin; the cadastre refine finds the village and its centroid.
    monkeypatch.setattr(geo, "geocode", lambda q: (11.30, 77.60))
    monkeypatch.setattr(vloc, "load_area_parcels_cached", lambda *a, **k: [{"sn": "1", "vc": "V"}])

    def _cands(parcels, surveys, anchor_utm, **kw):
        # centre chosen so the round-trip back to lat/lon is well inside Tamil Nadu
        return [{"vc": "933291", "center": (787866, 1254375), "n_overlap": 11,
                 "source": _Src({})}]
    monkeypatch.setattr(vloc, "village_candidates", _cands)

    r = CoordinateFinderAgent().find("NASIYANUR", {"110", "111"}, district="Erode", taluk="Erode")
    assert r["confidence"] == "high" and r["method"] == "tngis-cadastre"
    assert r["vc"] == "933291" and r["n_surveys"] == 11
    assert 11.2 < r["lat"] < 11.5 and 77.5 < r["lon"] < 77.8      # exact village centroid


def test_geocode_only_is_medium_fallback(monkeypatch):
    monkeypatch.setattr(geo, "geocode", lambda q: (11.2812, 77.5896))
    monkeypatch.setattr(vloc, "load_area_parcels_cached", lambda *a, **k: [])
    monkeypatch.setattr(vloc, "village_candidates", lambda *a, **k: [])   # no cadastre match

    r = CoordinateFinderAgent().find("KANDAMPALAYAM", {"9", "10"}, district="Erode",
                                     taluk="Perundurai")
    assert r["confidence"] == "medium" and r["method"].startswith("geocode:")
    assert r["lat"] == 11.2812 and r["lon"] == 77.5896


def test_nothing_found_is_reported(monkeypatch):
    monkeypatch.setattr(geo, "geocode", lambda q: None)               # nothing geocodes
    r = CoordinateFinderAgent().find("NOWHERE", {"1"}, district="", taluk="")
    assert r["confidence"] == "none" and r["lat"] is None


def test_cadastre_failure_falls_back_to_geocode(monkeypatch):
    # if the cadastre refine raises (e.g. parquet missing), the agent still returns the geocode.
    monkeypatch.setattr(geo, "geocode", lambda q: (11.34, 77.64))

    def _boom(*a, **k):
        raise FileNotFoundError("no parquet")
    monkeypatch.setattr(vloc, "load_area_parcels_cached", _boom)

    r = CoordinateFinderAgent().find("X", {"1"}, district="Erode", taluk="Erode")
    assert r["confidence"] == "medium" and r["lat"] == 11.34
