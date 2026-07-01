"""Tests for the M5 cadastral ingest + fit (file sources; S3 path tested live)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m2_georef"))
from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf
from landintel.pipeline.m5_cadastral.source import load_cadastral, VectorFileCadastralSource, _norm_survey
from landintel.pipeline.m5_cadastral.fit import fit_plot_to_parcel


def test_norm_survey():
    assert _norm_survey("82/1") == "82"
    assert _norm_survey("784A") == "784"
    assert _norm_survey("xx") is None


def _write_geojson(path, survey, verts_utm):
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"survey_no": survey},
        "geometry": {"type": "Polygon", "coordinates": [[[float(x), float(y)] for x, y in verts_utm] + [[float(verts_utm[0][0]), float(verts_utm[0][1])]]]}}]}
    Path(path).write_text(json.dumps(fc))


def test_geojson_source_and_fit(m1_dxf, tmp_path):
    from conftest import PLOT_VERTS, apply_true_transform
    utm = apply_true_transform(np.array(PLOT_VERTS))
    gj = tmp_path / "cad.geojson"
    _write_geojson(gj, "784", utm)
    src = load_cadastral(gj, source_crs="EPSG:32643")
    assert src.survey_numbers() == {"784"}
    parcel = src.get("784")
    assert parcel is not None and parcel.polygon.area > 100

    m1 = extract_m1_dxf(m1_dxf)
    fit = fit_plot_to_parcel(m1, parcel)
    assert fit is not None
    # placed corner ring should land on the true UTM parcel
    ring = fit.adjusted[[s.index for s in m1.stones]][:4]
    for e in utm:
        assert np.min(np.linalg.norm(ring - e, axis=1)) < 5.0


def test_unknown_survey_returns_none(m1_dxf, tmp_path):
    from conftest import PLOT_VERTS, apply_true_transform
    gj = tmp_path / "c.geojson"
    _write_geojson(gj, "999", apply_true_transform(np.array(PLOT_VERTS)))
    src = load_cadastral(gj, source_crs="EPSG:32643")
    assert src.get("784") is None


def _ring(cx, cy, side):
    h = side / 2.0
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


def test_recovered_candidates_returns_alternative_rings(tmp_path):
    """A survey appearing as TWO rings: get() = largest; recovery = the rest."""
    big = _ring(700000, 1200000, 40.0)     # area 1600
    small = _ring(700100, 1200000, 20.0)   # area 400
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"survey_no": "55"},
         "geometry": {"type": "Polygon", "coordinates": [[*[[float(x), float(y)] for x, y in big], [float(big[0][0]), float(big[0][1])]]]}},
        {"type": "Feature", "properties": {"survey_no": "55"},
         "geometry": {"type": "Polygon", "coordinates": [[*[[float(x), float(y)] for x, y in small], [float(small[0][0]), float(small[0][1])]]]}},
    ]}
    gj = tmp_path / "multi.geojson"
    Path(gj).write_text(json.dumps(fc))
    src = load_cadastral(gj, source_crs="EPSG:32643")

    primary = src.get("55")
    assert primary is not None and abs(primary.polygon.area - 1600) < 1.0   # largest kept
    cands = src.recovered_candidates("55")
    assert len(cands) == 1 and abs(cands[0].polygon.area - 400) < 1.0        # smaller offered
    assert all(c.polygon.area <= primary.polygon.area for c in cands)        # largest-first


def test_recovered_candidates_empty_for_clean_parcel(tmp_path):
    from conftest import PLOT_VERTS, apply_true_transform
    gj = tmp_path / "clean.geojson"
    _write_geojson(gj, "784", apply_true_transform(np.array(PLOT_VERTS)))
    src = load_cadastral(gj, source_crs="EPSG:32643")
    assert src.recovered_candidates("784") == []        # single ring -> no extras
    assert src.recovered_candidates("999") == []        # unknown survey -> empty, no crash
