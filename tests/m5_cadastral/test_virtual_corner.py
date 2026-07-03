"""3-corner (triangle) rigid-fit regression.

The proposed "virtual 4th corner" augmentation for 3-corner umeyama seeds was
implemented, measured, and REJECTED (see the note in m5_cadastral/fit.py): a derived
corner adds zero information and its reflected noise forms a torque couple that
WORSENS worst-case rotation error. These tests lock the behaviour that actually
carries 3-corner plots -- the existing seed + ICP + IoU tie-break path -- so
triangular plots keep fitting rigidly with geometry preserved. Synthetic only.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from landintel.pipeline.m5_cadastral.fit import _rigid_from_parcel, fit_plot_to_parcel
from landintel.pipeline.m5_cadastral.source import CadastralParcel


def _similarity(pts, angle_deg, s, t):
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return s * (np.asarray(pts, float) @ R.T) + np.asarray(t, float)


def _lens(r):
    n = len(r)
    return np.array([np.linalg.norm(r[(i + 1) % n] - r[i]) for i in range(n)])


def test_rigid_from_parcel_triangle_recovers_pose():
    # A triangular plot against its rotated+translated parcel: the fit must land
    # on the parcel with scale ~1 and a small residual.
    ring = np.array([[0.0, 0.0], [70.0, 0.0], [20.0, 45.0]])
    placed_truth = _similarity(ring, angle_deg=52.0, s=1.0, t=(781000.0, 1253000.0))
    parcel = CadastralParcel("T1", Polygon([tuple(p) for p in placed_truth]))

    fit = _rigid_from_parcel(ring, parcel)
    assert fit is not None
    R, s, t, resid = fit
    assert 0.5 < s < 2.0
    assert resid < 5.0
    # similarity transform preserves the FMB shape: edge-length RATIOS unchanged
    placed = s * (ring @ R.T) + t
    orig, new = _lens(ring), _lens(placed)
    assert np.allclose(new / new.sum(), orig / orig.sum(), atol=1e-6)


def test_rigid_from_parcel_triangle_survives_one_jittered_parcel_corner():
    # The raster/vector parcel corner can be a few metres off; the ICP + IoU path
    # must still recover approximately the right pose (position within metres).
    ring = np.array([[0.0, 0.0], [80.0, 0.0], [25.0, 40.0]])
    truth = _similarity(ring, angle_deg=-23.0, s=1.0, t=(780500.0, 1252500.0))
    noisy = truth.copy()
    noisy[2] += (4.0, -3.0)                              # one bad corner (5 m off)
    parcel = CadastralParcel("T2", Polygon([tuple(p) for p in noisy]))

    fit = _rigid_from_parcel(ring, parcel)
    assert fit is not None
    R, s, t, _resid = fit
    placed = s * (ring @ R.T) + t
    centroid_err = float(np.linalg.norm(placed.mean(axis=0) - truth.mean(axis=0)))
    assert centroid_err < 5.0                            # position holds under jitter
