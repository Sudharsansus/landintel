"""M4 village deliverable -- report straight from M2 GeorefResult dispositions.

A corridor run produces GeorefResults + a combined DXF; this builds the shippable
village PDF + Excel + delivery zip from them (areas from each placed plot's verified
boundary). Tested with duck-typed result rows and a real ACCEPT plot DXF so the
area on the report matches the geometry.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass

import ezdxf

from landintel.core.enums import LayerType
from landintel.pipeline.m4_report.village import (build_village_delivery,
                                                  rows_from_results,
                                                  village_area_statement_pdf,
                                                  village_excel)


@dataclass
class FakeResult:
    survey_number: str
    recommendation: str
    output_file: str = ""
    chain_coverage: float = 0.0
    field_residual_max: float = float("inf")
    match_method: str = ""


def _square_dxf(path, side=100.0):
    lyr = LayerType.BOUNDARY.value
    doc = ezdxf.new()
    doc.layers.add(lyr, color=7)
    doc.modelspace().add_lwpolyline(
        [(0, 0), (side, 0), (side, side), (0, side), (0, 0)], dxfattribs={"layer": lyr})
    doc.saveas(str(path))
    return str(path)


def test_rows_from_results_areas_only_for_placed(tmp_path):
    placed_dxf = _square_dxf(tmp_path / "724.dxf", 100.0)        # 1 ha
    results = [
        FakeResult("724", "ACCEPT", placed_dxf, 0.82, 0.003, "geometric"),
        FakeResult("999", "NO_COVERAGE", "", 0.10, float("inf")),
        FakeResult("500", "REVIEW", "", 0.40, float("inf")),
    ]
    rows = rows_from_results(results)
    by_sn = {r.survey_number: r for r in rows}
    assert abs(by_sn["724"].area_ha - 1.0) < 1e-3 and by_sn["724"].georeferenced
    assert by_sn["999"].area_ha is None and not by_sn["999"].georeferenced
    assert by_sn["500"].area_ha is None
    # placed rows sort first
    assert rows[0].survey_number == "724"


def test_pdf_and_excel_generate_bytes(tmp_path):
    results = [FakeResult("724", "ACCEPT", _square_dxf(tmp_path / "p.dxf"), 0.82, 0.003)]
    rows = rows_from_results(results)
    pdf = village_area_statement_pdf(rows, village="INGUR", crs="EPSG:32643")
    xlsx = village_excel(rows)
    assert pdf[:4] == b"%PDF" and len(pdf) > 800
    assert xlsx[:2] == b"PK" and len(xlsx) > 400        # xlsx is a zip


def test_delivery_zip_bundles_reports_and_dxf(tmp_path):
    combined = tmp_path / "combined_village.dxf"
    _square_dxf(combined, 120.0)
    results = [FakeResult("724", "ACCEPT", _square_dxf(tmp_path / "724.dxf"), 0.9, 0.01)]
    out = build_village_delivery(results, combined, tmp_path / "deliver",
                                 village="INGUR", crs="EPSG:32643")
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "village_area_statement.pdf" in names
    assert "village_area_breakdown.xlsx" in names
    assert "combined_village.dxf" in names
    assert (tmp_path / "deliver" / "village_area_statement.pdf").exists()
