"""Regression tests for the corridor land-schedule IDENTITY gate.

Geometry alone cannot resolve plot identity on a dense corridor: congruent plot
shapes match the wrong seat (measured: 13 false positives on real INGUR data).
The transmission-line land schedule lists which survey numbers the corridor
actually crosses; gating matching to that set removes the false positives. A plot
that PASSES the gate is identity-confirmed, so a weak auto-placement becomes
REVIEW (human seeds it), never a silent NO_COVERAGE drop.
"""

from __future__ import annotations

import ezdxf
import pytest

from landintel.pipeline.m2_georef.extract_surveyor import (
    extract_corridor_surveys,
    extract_surveyor,
)
from landintel.pipeline.m2_georef.pipeline import georef_single


# --- extract_corridor_surveys ----------------------------------------------


def test_extract_corridor_surveys_reads_survey_number_layer(tmp_path):
    doc = ezdxf.new("R2010")
    doc.layers.add("SURVEY NUMBER")
    msp = doc.modelspace()
    for t in ("667", "773", "82/1", "Y8t6", "725"):  # last-but-one is OCR noise
        e = msp.add_text(t, dxfattribs={"layer": "SURVEY NUMBER"})
        e.set_placement((0, 0))
    # A label on another layer must be ignored.
    msp.add_text("999", dxfattribs={"layer": "0"}).set_placement((0, 0))
    p = tmp_path / "schedule.dxf"
    doc.saveas(p)

    surveys = extract_corridor_surveys(p)
    assert surveys == {"667", "773", "82", "725"}   # "82/1" -> "82"; "Y8t6" rejected
    assert "999" not in surveys                       # wrong layer ignored


def test_extract_corridor_surveys_missing_layer_returns_empty(tmp_path):
    doc = ezdxf.new("R2010")
    doc.modelspace().add_text("123").set_placement((0, 0))
    p = tmp_path / "no_layer.dxf"
    doc.saveas(p)
    assert extract_corridor_surveys(p) == set()


# --- the gate in georef_single ---------------------------------------------


def test_gate_excludes_off_schedule_without_matching(m1_dxf, surveyor_dxf, tmp_path):
    """A plot whose survey number is NOT on the schedule is NO_COVERAGE and is
    never matched (so it can never become a geometric false positive)."""
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    r = georef_single(m1_dxf, surveyor, tmp_path / "out",
                      corridor_surveys={"999", "1000"})  # synthetic plot is "784"
    assert r.survey_number == "784"
    assert r.recommendation == "NO_COVERAGE"
    assert not r.matched
    assert not r.output_file            # nothing written -> not clubbed downstream
    assert "not on the corridor" in r.error


def test_gate_keeps_on_schedule_plot(m1_dxf, surveyor_dxf, tmp_path):
    """A plot ON the schedule passes the gate and is placed (ACCEPT/REVIEW),
    never dropped as NO_COVERAGE -- identity is confirmed."""
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    r = georef_single(m1_dxf, surveyor, tmp_path / "out2",
                      corridor_surveys={"784"})
    assert r.matched
    assert r.recommendation in ("ACCEPT", "REVIEW")
    assert r.recommendation != "NO_COVERAGE"


def test_confirmed_corridor_weak_match_is_review_not_nocoverage(
        surveyor_dxf, tmp_path):
    """A schedule-confirmed plot that cannot auto-match (too few stones) is
    surfaced as REVIEW for human seeding -- a real corridor plot is never lost."""
    from landintel.core.models import Boundary, CornerPoint, Plot
    from landintel.pipeline.m1_extract.to_dxf import write_dxf
    verts = [(0.0, 0.0), (3.3, 0.0), (1.4, 2.9)]   # tiny triangle, won't congruently match
    plot = Plot(client_id="c", survey_no="667", district="E", taluk="P",
                village="INGUR", scale=1000, stated_area=0.001,
                boundary=Boundary(points=verts + [verts[0]]),
                corner_points=[CornerPoint(label=str(i), x=x, y=y)
                               for i, (x, y) in enumerate(verts)])
    odd = write_dxf(plot, tmp_path / "m1_667.dxf")
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    r = georef_single(odd, surveyor, tmp_path / "out3", corridor_surveys={"667"})
    assert r.recommendation == "REVIEW"        # on corridor -> never NO_COVERAGE
    assert r.recommendation != "NO_COVERAGE"
