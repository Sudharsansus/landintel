"""UTM auto-zone detection -- the engine must pick the right CRS for ANY land."""
from __future__ import annotations

from landintel.pipeline.utm import (utm_crs_for_lon, utm_crs_for_wgs84_point,
                                    detect_crs_from_cadastral)


def test_utm_zone_for_tamil_nadu_split():
    assert utm_crs_for_lon(77.6) == "EPSG:32643"   # INGUR/Erode -> 43N (west of 78E)
    assert utm_crs_for_lon(78.5) == "EPSG:32644"   # east of 78E  -> 44N
    assert utm_crs_for_lon(78.0) == "EPSG:32644"   # boundary -> 44N
    assert utm_crs_for_lon(72.8) == "EPSG:32643"   # Mumbai-ish still 43N edge
    assert utm_crs_for_wgs84_point(77.6, 11.2) == "EPSG:32643"


def test_detect_crs_from_geojson(tmp_path):
    gj = tmp_path / "parcel.geojson"
    gj.write_text('{"type":"FeatureCollection","features":[{"type":"Feature",'
                  '"properties":{"survey_no":"724"},"geometry":{"type":"Polygon",'
                  '"coordinates":[[[78.6,11.2],[78.7,11.2],[78.7,11.3],[78.6,11.2]]]}}]}')
    assert detect_crs_from_cadastral(gj) == "EPSG:32644"   # 78.6E -> 44N, auto-detected


def test_detect_crs_returns_none_for_landxml(tmp_path):
    # LandXML is already projected (not lon/lat) -> no zone sniff, keep explicit CRS
    x = tmp_path / "s.xml"
    x.write_text("<LandXML><CgPoints/></LandXML>")
    assert detect_crs_from_cadastral(x) is None
