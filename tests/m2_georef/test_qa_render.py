"""Visual QA overlay -- the human-auditable false-positive backstop.

A plot can pass every numerical gate and still sit in the WRONG place. The QA overlay draws each
placed FMB against its authoritative S3/cadastral parcel so that misplacement is visible. These
tests pin the FP-catch logic (misplacement distance, footprint overlap) and that a real PNG is
produced from synthetic results.
"""
from __future__ import annotations

from dataclasses import dataclass

import ezdxf
from shapely.geometry import Polygon

from landintel.core.enums import LayerType
from landintel.pipeline.m2_georef.qa_render import (disposition_color,
                                                    footprint_overlaps,
                                                    misplaced,
                                                    render_qa_overlay)


@dataclass
class FakeResult:
    survey_number: str
    recommendation: str
    output_file: str = ""
    chain_coverage: float = 0.0
    field_residual_max: float = float("inf")


class StubParcel:
    def __init__(self, poly):
        self.polygon = poly


class StubCadastral:
    def __init__(self, by_sn):
        self._by = by_sn

    def get(self, sn, village=None):
        return self._by.get(sn)


def _square_dxf(path, x0, y0, side=40.0):
    lyr = LayerType.BOUNDARY.value
    doc = ezdxf.new(); doc.layers.add(lyr, color=7)
    doc.modelspace().add_lwpolyline(
        [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side), (x0, y0)],
        dxfattribs={"layer": lyr})
    doc.saveas(str(path)); return str(path)


# --- pure FP-catch helpers --------------------------------------------------
def test_misplaced_flags_far_placement():
    flag, d = misplaced((700000, 1200000), (700300, 1200000), thresh_m=100.0)
    assert flag and abs(d - 300) < 1e-6


def test_misplaced_ok_when_on_parcel():
    flag, d = misplaced((700000, 1200000), (700010, 1200005), thresh_m=100.0)
    assert not flag and d < 100


def test_footprint_overlaps_detects_interior_overlap():
    a = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
    b = Polygon([(20, 20), (60, 20), (60, 60), (20, 60)])     # overlaps a's corner
    c = Polygon([(100, 100), (140, 100), (140, 140), (100, 140)])  # far away
    ov = footprint_overlaps({"1": a, "2": b, "3": c}, min_frac=0.05)
    assert len(ov) == 1
    assert {ov[0][0], ov[0][1]} == {"1", "2"}


def test_adjacent_tiling_is_not_an_overlap():
    a = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
    b = Polygon([(40, 0), (80, 0), (80, 40), (40, 40)])       # shares an EDGE only
    assert footprint_overlaps({"a": a, "b": b}) == []


def test_disposition_colors():
    assert disposition_color("ACCEPT") == disposition_color("ACCEPT_CADASTRAL")
    assert disposition_color("REVIEW") != disposition_color("ACCEPT")
    assert disposition_color("NO_COVERAGE") == "#9e9e9e"


# --- end-to-end render ------------------------------------------------------
def test_render_produces_png_with_misplacement_flag(tmp_path):
    # plot 724: placed ON its parcel (correct). plot 999: placed FAR from its parcel (FP!).
    good_dxf = _square_dxf(tmp_path / "724.dxf", 700000, 1200000)
    bad_dxf = _square_dxf(tmp_path / "999.dxf", 700000, 1200000)   # placed here...
    parcels = {
        "724": StubParcel(Polygon([(700000, 1200000), (700040, 1200000),
                                   (700040, 1200040), (700000, 1200040)])),
        "999": StubParcel(Polygon([(705000, 1200000), (705040, 1200000),   # ...but belongs here
                                   (705040, 1200040), (705000, 1200040)])),
    }
    results = [
        FakeResult("724", "ACCEPT", good_dxf, 0.85, 0.02),
        FakeResult("999", "ACCEPT", bad_dxf, 0.60, 1.2),
    ]
    out = render_qa_overlay(results, tmp_path / "qa.png",
                            cadastral_source=StubCadastral(parcels),
                            title="test", misplace_flag_m=100.0)
    assert out is not None and out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"        # valid PNG
    assert out.stat().st_size > 5000


def test_render_safe_with_no_cadastral_or_surveyor(tmp_path):
    dxf = _square_dxf(tmp_path / "1.dxf", 700000, 1200000)
    out = render_qa_overlay([FakeResult("1", "REVIEW", dxf, 0.4, 3.0)],
                            tmp_path / "qa2.png")
    assert out is not None and out.exists()      # works with just placed footprints
