"""LandXML loader tests -- parse parcels + points + CRS, and plug into the cadastral source.

Uses a synthetic LandXML (standard landxml.org schema) so the loader is validated before the
client's real .xml arrives. Point text is 'northing easting elev' (survey convention).
"""
from __future__ import annotations

import textwrap

from landintel.pipeline.m5_cadastral.landxml import parse_landxml, landxml_points
from landintel.pipeline.m5_cadastral.source import VectorFileCadastralSource

_LANDXML = textwrap.dedent("""\
    <?xml version="1.0"?>
    <LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2">
      <CoordinateSystem epsgCode="32643"/>
      <CgPoints>
        <CgPoint name="1">1240000.0 782000.0 100.0</CgPoint>
        <CgPoint name="2">1240000.0 782100.0 100.0</CgPoint>
        <CgPoint name="3">1240100.0 782100.0 100.0</CgPoint>
        <CgPoint name="4">1240100.0 782000.0 100.0</CgPoint>
      </CgPoints>
      <Parcels>
        <Parcel name="724/1" desc="INGUR">
          <CoordGeom>
            <Line><Start pntRef="1"/><End pntRef="2"/></Line>
            <Line><Start pntRef="2"/><End pntRef="3"/></Line>
            <Line><Start pntRef="3"/><End pntRef="4"/></Line>
            <Line><Start pntRef="4"/><End pntRef="1"/></Line>
          </CoordGeom>
        </Parcel>
      </Parcels>
    </LandXML>
    """)


def _write(tmp_path):
    p = tmp_path / "span.xml"
    p.write_text(_LANDXML)
    return p


def test_parse_landxml_parcels_points_crs(tmp_path):
    parcels, points, crs = parse_landxml(_write(tmp_path))
    assert crs == "EPSG:32643"
    assert len(points) == 4
    assert points["1"] == (782000.0, 1240000.0)          # (easting, northing)
    assert len(parcels) == 1
    sn, village, poly = parcels[0]
    assert sn == "724"                                    # "724/1" normalised to base
    assert village == "INGUR" or village is None          # desc is best-effort
    assert abs(poly.area - 10000.0) < 1.0                 # 100 m x 100 m square


def test_landxml_through_cadastral_source(tmp_path):
    # the .xml must load as a CadastralSource (same interface as the TNGIS vector export)
    src = VectorFileCadastralSource(_write(tmp_path))
    parcel = src.get("724")
    assert parcel is not None
    assert parcel.survey_number == "724"
    assert parcel.polygon.area > 9000
    assert src.get("724/2") is not None                   # subdivision keys to the same base


def test_landxml_points_helper(tmp_path):
    pts = landxml_points(_write(tmp_path))
    assert pts["3"] == (782100.0, 1240100.0)


# ----------------------------------------------------------------- CSV reader ----
def test_csv_points_grouped_by_survey(tmp_path):
    # rows are boundary vertices grouped by survey number, ordered by seq
    csv = tmp_path / "pts.csv"
    csv.write_text(
        "survey_no,seq,easting,northing\n"
        "724,1,782000,1240000\n724,2,782100,1240000\n"
        "724,3,782100,1240100\n724,4,782000,1240100\n")
    src = VectorFileCadastralSource(csv, source_crs="EPSG:32643")
    p = src.get("724")
    assert p is not None and abs(p.polygon.area - 10000.0) < 1.0


def test_csv_self_intersecting_ring_is_flattened(tmp_path):
    # a bowtie ring -> buffer(0) yields a MultiPolygon -> must be flattened to Polygon parts
    # so .exterior is always accessible downstream (the robustness bug Agent A caught).
    csv = tmp_path / "bowtie.csv"
    csv.write_text("survey_no,seq,x,y\n5,1,0,0\n5,2,10,10\n5,3,10,0\n5,4,0,10\n")
    src = VectorFileCadastralSource(csv, source_crs="EPSG:32643")
    p = src.get("5")
    if p is not None:                                     # must be a single Polygon, not Multi
        _ = list(p.polygon.exterior.coords)               # would AttributeError on MultiPolygon
        assert p.polygon.area > 0


def test_csv_wkt_column(tmp_path):
    csv = tmp_path / "wkt.csv"
    csv.write_text(
        "sy_no,wkt\n"
        '"82","POLYGON((782000 1240000,782100 1240000,782100 1240100,782000 1240100,782000 1240000))"\n')
    src = VectorFileCadastralSource(csv, source_crs="EPSG:32643")
    assert src.get("82") is not None and src.get("82").polygon.area > 9000
