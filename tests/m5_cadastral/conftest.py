"""Shared fixtures for M2 georeferencing tests.

The synthetic M1 DXF is built through the REAL ``m1_extract.to_dxf`` serializer
from a hand-specified ``Plot`` -- so what M2 parses is byte-for-byte the layer
convention M1 actually emits (BOUNDARY edges, STONES TEXT, etc.), not a
hand-rolled approximation that could drift.

The synthetic surveyor DXF is built directly with ezdxf to mirror the INGUR
tower-corridor convention: POINT entities on layer '0', co-located Point_Code
TEXT, and an OPEN SITE DATA LINE polyline tracing the boundary stones. A known
similarity transform (scale, rotation, translation into UTM Zone 44N) maps the
plot's relative metres onto surveyor coordinates, so the expected georeferencing
result is known exactly.
"""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import numpy as np
import pytest

from landintel.core.models import Boundary, CornerPoint, Plot
from landintel.pipeline.m1_extract.to_dxf import write_dxf

# Real INGUR surveyor DXF (present on this machine; tests that need it skip if absent).
REAL_SURVEYOR_DXF = (
    Path(__file__).resolve().parents[2] / "test2" / "INGUR" / "INGUR RAW DATA FILE.dxf"
)

# A quadrilateral plot with four DISTINCT edge lengths -> a discriminating
# fingerprint. Relative metres, as M1 produces.
#   P1(0,0) -> P2(50,0) -> P3(50,35) -> P4(0,22) -> close
#   edges: 50.00, 35.00, 51.66, 22.00   (perimeter 158.66)
PLOT_VERTS = [(0.0, 0.0), (50.0, 0.0), (50.0, 35.0), (0.0, 22.0)]
STONE_LABELS = ["1", "2", "3", "4"]

# Known similarity transform applied to build surveyor (UTM) coordinates.
TRUE_SCALE = 1.0
TRUE_THETA_DEG = 20.0
TRUE_T = np.array([783000.0, 1241000.0])


def _R(theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def apply_true_transform(pts: np.ndarray) -> np.ndarray:
    """Map relative-metre points to surveyor UTM coords via the known transform."""
    return TRUE_SCALE * (pts @ _R(TRUE_THETA_DEG).T) + TRUE_T


@pytest.fixture
def synthetic_plot() -> Plot:
    """A 4-corner plot exercising the boundary + stones layers M2 reads."""
    ring = PLOT_VERTS + [PLOT_VERTS[0]]
    return Plot(
        client_id="client_test",
        survey_no="784",
        district="Erode",
        taluk="Perundurai",
        village="INGUR",
        scale=1000,
        stated_area=0.1,
        boundary=Boundary(points=ring),
        corner_points=[
            CornerPoint(label=lbl, x=x, y=y)
            for lbl, (x, y) in zip(STONE_LABELS, PLOT_VERTS)
        ],
    )


@pytest.fixture
def m1_dxf(synthetic_plot: Plot, tmp_path: Path) -> Path:
    """Write the synthetic plot to a real M1 DXF and return its path."""
    return write_dxf(synthetic_plot, tmp_path / "m1_synth_784.dxf")


def build_surveyor_dxf(
    path: Path,
    *,
    include_decoy: bool = True,
) -> Path:
    """Build a synthetic surveyor DXF in the INGUR convention.

    Target plot stones get indices 0..3 (created first); optional decoy stones
    (a triangle with very different edge lengths, placed far away) get the later
    indices so they force the matcher to discriminate rather than match trivially.
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("Point_Code")
    doc.layers.add("SITE DATA LINE")

    surv_pts = apply_true_transform(np.array(PLOT_VERTS))

    def add_stone(x: float, y: float, code: str = "B") -> None:
        msp.add_point((x, y), dxfattribs={"layer": "0"})
        t = msp.add_text(code, dxfattribs={"layer": "Point_Code", "height": 1.0})
        t.set_placement((x, y))

    # Target stones first -> indices 0..3 in extraction order.
    for x, y in surv_pts:
        add_stone(float(x), float(y))

    # OPEN SITE DATA LINE tracing the 4 stones around the ring (4 edges, window N).
    trace = [tuple(surv_pts[i]) for i in range(4)] + [tuple(surv_pts[0])]
    msp.add_lwpolyline(trace, dxfattribs={"layer": "SITE DATA LINE"})

    if include_decoy:
        # A decoy triangle with clearly different edge lengths, far away.
        decoy = np.array([(900.0, 900.0), (1000.0, 905.0), (940.0, 1010.0)])
        decoy = apply_true_transform(decoy)
        for x, y in decoy:
            add_stone(float(x), float(y))
        dtrace = [tuple(decoy[i]) for i in range(3)] + [tuple(decoy[0])]
        msp.add_lwpolyline(dtrace, dxfattribs={"layer": "SITE DATA LINE"})

    doc.saveas(path)
    return path


@pytest.fixture
def surveyor_dxf(tmp_path: Path) -> Path:
    """Write the synthetic surveyor DXF and return its path."""
    return build_surveyor_dxf(tmp_path / "surveyor_synth.dxf")
