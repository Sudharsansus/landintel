"""Rotation-fill disambiguation in the cadastral rigid fit (fit.py).

The bug: corner residual alone is rotation-ambiguous on a NEAR-SQUARE plot/parcel
(the orientation gate is skipped there), so ``_rigid_from_parcel`` could lock onto a
~90-degrees-off pose that has a marginally lower corner residual but FILLS the parcel
poorly. The fix adds footprint IoU against the parcel as a selection criterion so the
rotation that best overlaps/fills the parcel wins near-ties in corner residual.

This test builds a near-square plot on a rectangular parcel whose skeleton has a small
asymmetry, so a 90-degrees-off pose can sneak a competitive corner residual. The correct
fill rotation must still be chosen. GENERAL geometry -- no survey numbers, no constants.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from landintel.pipeline.m2_georef.extract_m1 import M1PlotData, M1Stone
from landintel.pipeline.m5_cadastral.fit import fit_plot_to_parcel, _is_near_square
from landintel.pipeline.m5_cadastral.source import CadastralParcel


def _m1_from_ring(ring):
    """Minimal M1PlotData whose outer corner ring is exactly ``ring`` (relative metres)."""
    stones = [M1Stone(x=float(x), y=float(y), label=str(i), index=i)
              for i, (x, y) in enumerate(ring)]
    m1 = M1PlotData(stones=stones, survey_number="TEST")
    m1.outer_stone_indices = list(range(len(ring)))
    return m1


def _parcel(coords):
    return CadastralParcel(survey_number="TEST", polygon=Polygon(coords),
                           village="V", source_crs="EPSG:32643")


def _iou(ring_a, poly_b):
    a = Polygon([(float(x), float(y)) for x, y in ring_a]).buffer(0)
    inter = a.intersection(poly_b).area
    union = a.area + poly_b.area - inter
    return inter / union if union > 0 else 0.0


def test_rotation_fill_preferred_over_lower_residual_flip():
    """A near-square plot on a near-square parcel: the well-FILLING rotation must win
    even though a ~90-degrees-off pose can score a competitive corner residual."""
    # Plot: a 40 x 34 quad (aspect ~1.18 -> near-square, orientation gate SKIPPED) with a
    # small chamfer on one corner so the four corners are distinguishable.
    plot = np.array([
        (0.0, 0.0),
        (40.0, 0.0),
        (40.0, 34.0),
        (8.0, 34.0),
        (0.0, 26.0),
    ])
    assert _is_near_square(plot), "this test requires the orientation gate to be skipped"

    # Parcel: the SAME shape, placed in UTM at its correct (0-degrees) orientation.
    t = np.array([700000.0, 1200000.0])
    parcel_coords = [tuple(p + t) for p in plot]
    parcel = _parcel(parcel_coords)

    m1 = _m1_from_ring([tuple(p) for p in plot])
    fit = fit_plot_to_parcel(m1, parcel)
    assert fit is not None and fit.method == "rigid"

    # The chosen pose must FILL the parcel (high IoU) and be near the correct orientation,
    # NOT a ~90-degrees-off low-residual lock.
    placed_ring = fit.adjusted[np.array(m1.outer_stone_indices)]
    iou = _iou(placed_ring, parcel.polygon)
    assert iou > 0.85, f"fit should fill the parcel (IoU>0.85), got {iou:.3f}"

    rot_deg = abs(math.degrees(math.atan2(fit.R[1, 0], fit.R[0, 0]))) % 360.0
    rot_deg = min(rot_deg, 360.0 - rot_deg)
    assert rot_deg < 25.0, f"fit should be near the correct rotation, got {rot_deg:.1f} deg"


def test_well_fit_plot_unchanged_by_iou_selection():
    """A clearly rectangular plot already fits at its best-residual pose with the best
    fill; the IoU criterion must be a no-op (still a clean, well-filling placement)."""
    plot = np.array([(0.0, 0.0), (60.0, 0.0), (60.0, 20.0), (0.0, 20.0)])
    t = np.array([700000.0, 1200000.0])
    th = math.radians(15.0)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    parcel_coords = [tuple(p @ R.T + t) for p in plot]
    parcel = _parcel(parcel_coords)

    m1 = _m1_from_ring([tuple(p) for p in plot])
    fit = fit_plot_to_parcel(m1, parcel)
    assert fit is not None and fit.method == "rigid"

    placed_ring = fit.adjusted[np.array(m1.outer_stone_indices)]
    iou = _iou(placed_ring, parcel.polygon)
    assert iou > 0.9, f"a well-fit rectangular plot should fill the parcel, got {iou:.3f}"
    assert 0.9 < fit.s < 1.1
