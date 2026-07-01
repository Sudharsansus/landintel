"""Tests for the deterministic M1->M2 DXF verification gate.

A good DXF (written by the real to_dxf serializer) must verify PROPER; the
failure modes the gate exists to stop -- collapsed boundary that excludes the
stones, duplicated boundary lines, an open ring, blown-up coordinates -- must
each be caught as a hard (gating) failure.
"""

from __future__ import annotations

from pathlib import Path

import ezdxf
import pytest

from landintel.core.enums import LayerType
from landintel.core.models import Boundary, CornerPoint, Plot
from landintel.pipeline.m1_extract.to_dxf import write_dxf
from landintel.pipeline.m1_extract.verify_dxf import verify_m1_dxf, write_verify_sidecar

# A square plot, 4 stones exactly on the corners.
_VERTS = [(0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (0.0, 30.0)]


def _good_plot() -> Plot:
    return Plot(
        client_id="c", survey_no="42", district="D", taluk="T", village="V",
        scale=1000, stated_area=0.12,  # 40*30 = 1200 m2 = 0.12 ha
        boundary=Boundary(points=_VERTS + [_VERTS[0]]),
        corner_points=[CornerPoint(label=str(i + 1), x=x, y=y)
                       for i, (x, y) in enumerate(_VERTS)],
    )


@pytest.fixture
def good_dxf(tmp_path: Path) -> Path:
    return write_dxf(_good_plot(), tmp_path / "good.dxf")


def test_good_dxf_is_proper(good_dxf: Path):
    r = verify_m1_dxf(good_dxf, stated_area_ha=0.12)
    assert r.proper, [c.detail for c in r.failures]
    assert not r.failures


def test_sidecar_written(good_dxf: Path):
    r = verify_m1_dxf(good_dxf, stated_area_ha=0.12)
    side = write_verify_sidecar(r)
    assert side.exists() and side.suffix == ".txt"
    assert "PROPER" in side.read_text(encoding="utf-8")


def test_collapsed_boundary_fails_on_area(good_dxf: Path, tmp_path: Path):
    """The 667-class collapse: boundary shrinks to a fraction of the true plot,
    so its area no longer matches the stated FMB area -> hard fail."""
    doc = ezdxf.readfile(str(good_dxf))
    msp = doc.modelspace()
    # Replace the boundary edges with a tiny 5x5 ring (collapsed perimeter).
    for e in list(msp.query("LWPOLYLINE")):
        if e.dxf.layer == LayerType.BOUNDARY.value:
            msp.delete_entity(e)
    tiny = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]
    for i in range(len(tiny) - 1):
        msp.add_lwpolyline([tiny[i], tiny[i + 1]],
                           dxfattribs={"layer": LayerType.BOUNDARY.value})
    out = tmp_path / "collapsed.dxf"
    doc.saveas(out)
    # True plot is 0.12 ha; collapsed boundary is 25 m2 = 0.0025 ha -> ~98% off.
    r = verify_m1_dxf(out, stated_area_ha=0.12)
    assert not r.proper
    assert any(c.name == "area_vs_stated" for c in r.failures)


def test_duplicate_boundary_line_fails(good_dxf: Path, tmp_path: Path):
    doc = ezdxf.readfile(str(good_dxf))
    msp = doc.modelspace()
    first = next(e for e in msp.query("LWPOLYLINE")
                 if e.dxf.layer == LayerType.BOUNDARY.value)
    pts = list(first.get_points())
    msp.add_lwpolyline([(p[0], p[1]) for p in pts],
                       dxfattribs={"layer": LayerType.BOUNDARY.value})
    out = tmp_path / "dup.dxf"
    doc.saveas(out)
    r = verify_m1_dxf(out)
    assert not r.proper
    assert any(c.name == "no_duplicate_boundary" for c in r.failures)


def test_open_boundary_fails(good_dxf: Path, tmp_path: Path):
    """Delete one boundary edge -> the ring no longer closes (degree-1 nodes)."""
    doc = ezdxf.readfile(str(good_dxf))
    msp = doc.modelspace()
    edges = [e for e in msp.query("LWPOLYLINE")
             if e.dxf.layer == LayerType.BOUNDARY.value]
    msp.delete_entity(edges[0])
    out = tmp_path / "open.dxf"
    doc.saveas(out)
    r = verify_m1_dxf(out)
    assert not r.proper
    assert any(c.name == "boundary_closed" for c in r.failures)


def test_wild_coordinates_fail(good_dxf: Path, tmp_path: Path):
    doc = ezdxf.readfile(str(good_dxf))
    msp = doc.modelspace()
    e = next(iter(msp.query("LWPOLYLINE")))
    e.set_points([(1e9, 1e9), (1e9 + 10, 1e9)])
    out = tmp_path / "wild.dxf"
    doc.saveas(out)
    r = verify_m1_dxf(out)
    assert not r.proper
    assert any(c.name == "coords_finite" for c in r.failures)


def test_area_mismatch_flags(good_dxf: Path):
    """A wildly wrong stated area is a hard fail; a small mismatch only warns."""
    r_bad = verify_m1_dxf(good_dxf, stated_area_ha=10.0)  # true is 0.12 ha
    area_check = next(c for c in r_bad.checks if c.name == "area_vs_stated")
    assert not area_check.passed and area_check.severity == "fail"
    assert not r_bad.proper


def test_label_noise_is_warn_not_gate(good_dxf: Path, tmp_path: Path):
    """A non-numeric dimension token must WARN, never block promotion."""
    doc = ezdxf.readfile(str(good_dxf))
    msp = doc.modelspace()
    msp.add_text("Y8t6", dxfattribs={"layer": LayerType.BOUNDARY_DIMENSIONS.value,
                                      "insert": (5, 5)})
    out = tmp_path / "noise.dxf"
    doc.saveas(out)
    r = verify_m1_dxf(out, stated_area_ha=0.12)
    assert r.proper  # still fit for M2
    assert any(c.name == "dimension_labels_numeric" for c in r.warnings)
