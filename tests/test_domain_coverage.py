"""Domain-coverage hardening from the TN-FMB research pass (2026-06-28).

Two verified, low-risk robustness gaps the research agent surfaced:
  1. verify.py Check 2 coordinate-range box was 44N-only and too tight -- it rejected
     valid parcels in southern TN (Kanyakumari ~8N) and eastern 44N districts. Now
     zone-agnostic across both TN UTM zones (43N + 44N).
  2. parse_header only read "Hect/Ares"; older sheets use Acre/Cent -> stated area was
     lost and the area cross-check silently switched off. Now parsed as a fallback.
"""
from __future__ import annotations

import ezdxf

from landintel.core.enums import LayerType
from landintel.pipeline.m1_extract.ocr import OCRDetection, parse_header
from landintel.pipeline.m2_georef.verify import verify_georef_dxf


# --- 1) zone-agnostic coordinate-range gate ---------------------------------

def _dxf_with_boundary(path, pts):
    doc = ezdxf.new()
    lyr = LayerType.BOUNDARY.value
    doc.layers.add(lyr, color=7)
    doc.modelspace().add_lwpolyline(list(pts) + [pts[0]], dxfattribs={"layer": lyr})
    doc.saveas(str(path))
    return str(path)


def _coord_check(path):
    res = verify_georef_dxf(path)
    return next(c for c in res.checks if c.name == "2_Coordinate_Range")


def test_southern_tn_43n_parcel_passes_range(tmp_path):
    # Kanyakumari-ish: low northing (~890k) that the old 1.1M floor wrongly rejected.
    pts = [(720000, 890000), (720050, 890000), (720050, 890050), (720000, 890050)]
    assert _coord_check(_dxf_with_boundary(tmp_path / "south.dxf", pts)).passed


def test_eastern_tn_44n_parcel_passes_range(tmp_path):
    # Eastern 44N (~80E): low easting (~420k) that the old 600k floor wrongly rejected.
    pts = [(420000, 1200000), (420050, 1200000), (420050, 1200050), (420000, 1200050)]
    assert _coord_check(_dxf_with_boundary(tmp_path / "east.dxf", pts)).passed


def test_ingur_43n_parcel_still_passes(tmp_path):
    pts = [(789000, 1240000), (789050, 1240000), (789050, 1240050), (789000, 1240050)]
    assert _coord_check(_dxf_with_boundary(tmp_path / "ingur.dxf", pts)).passed


def test_relative_metre_origin_still_rejected(tmp_path):
    # Un-georeferenced M1 coords near the origin must still FAIL the range gate.
    pts = [(0, 0), (50, 0), (50, 50), (0, 50)]
    assert not _coord_check(_dxf_with_boundary(tmp_path / "rel.dxf", pts)).passed


# --- 2) Acre/Cent area parsing fallback -------------------------------------

def _det(text):
    return OCRDetection(text=text, confidence=0.95,
                        polygon=((0, 0), (1, 0), (1, 1), (0, 1)))


def test_hect_ares_still_parsed():
    h = parse_header([_det("Area : Hect 01 Ares 66.50")])
    assert abs(h.stated_area_ha - 1.665) < 1e-9


def test_acre_cent_fallback_parsed():
    # 2 acres 50 cents = 2.5 acres = 2.5 * 0.404686 = 1.011715 ha
    h = parse_header([_det("Area : Acre 02 Cent 50.00")])
    assert abs(h.stated_area_ha - 2.5 * 0.404686) < 1e-6


def test_hect_takes_precedence_over_acre():
    # If both forms appear, the modern Hect/Ares wins (parsed first).
    h = parse_header([_det("Hect 01 Ares 00.00"), _det("Acre 99 Cent 00")])
    assert abs(h.stated_area_ha - 1.0) < 1e-9


def test_no_area_token_leaves_none():
    h = parse_header([_det("District : Sivagangai")])
    assert h.stated_area_ha is None


def test_decimal_hectare_form_parsed():
    h = parse_header([_det("Area : 1.665 Ha")])
    assert abs(h.stated_area_ha - 1.665) < 1e-9


def test_decimal_acre_form_parsed():
    h = parse_header([_det("Area : 2.5 acres")])
    assert abs(h.stated_area_ha - 2.5 * 0.404686) < 1e-6


def test_decimal_cents_form_parsed():
    h = parse_header([_det("Area : 80 cents")])
    assert abs(h.stated_area_ha - 80 * (0.404686 / 100.0)) < 1e-6


def test_generic_form_does_not_grab_unrelated_numbers():
    # "Survey No : 252" must NOT be read as an area (no unit, not the Area field).
    h = parse_header([_det("Survey No : 252"), _det("Scale : 1 : 2021")])
    assert h.stated_area_ha is None
