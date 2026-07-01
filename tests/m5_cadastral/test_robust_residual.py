"""Robust corner residual (adopted M2 "diamond" #13/#15: partial-Chamfer / Modified-Hausdorff).

The plain ``rot_residual`` is a MEAN nearest-neighbour distance from the placed M1 corners to
the raster-traced parcel boundary. A single OCR/raster-jittered corner inflates that mean and
can fail an otherwise-correct fit ONLY on residual. ``_robust_corner_residual`` trims the worst
fraction so the fit-quality estimate shrugs off one bad corner -- strictly recall-additive, and
0-FP-safe because seat-locality (not residual) is the false-positive lock.

GENERAL geometry -- no survey numbers, no INGUR-specific constants.
"""
from __future__ import annotations

import types

import numpy as np
from shapely.geometry import Polygon

from landintel.pipeline.m2_georef.extract_m1 import M1PlotData, M1Stone
from landintel.pipeline.m5_cadastral.fit import (
    _densify_ring,
    _robust_corner_residual,
    fit_plot_to_parcel,
)
from landintel.pipeline.m5_cadastral.source import CadastralParcel

import importlib
# NB: m2_club/__init__ re-exports the ``cadastral_seat`` *function*, shadowing the module
# attribute, so fetch the module object itself for monkeypatching its module-level flag.
cadastral_seat = importlib.import_module("landintel.pipeline.m2_club.cadastral_seat")


def _square_boundary(side=40.0, step=1.0):
    coords = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
    return _densify_ring(coords), coords


def test_robust_equals_mean_on_clean_fit():
    """Corners exactly on the boundary: nothing to trim, robust == plain mean (~0)."""
    boundary, coords = _square_boundary()
    placed = np.array(coords, float)          # the four corners, on the boundary
    d, _ = np.zeros(len(placed)), None
    mean = float(np.mean([np.min(np.hypot(*(boundary - p).T)) for p in placed]))
    robust = _robust_corner_residual(placed, boundary)
    assert robust < 0.6                        # essentially zero (densify step granularity)
    assert abs(robust - mean) < 0.6            # tracks the plain mean when the fit is clean


def test_robust_ignores_one_jittered_corner():
    """One corner thrown 10 m off the boundary: the plain mean jumps, robust stays low."""
    boundary, coords = _square_boundary(side=40.0)
    placed = np.array(coords, float)
    placed[2] = (40.0, 50.0)                    # push one corner 10 m past the top edge
    # plain forward mean ~ (0 + 0 + 10 + 0) / 4 = 2.5
    mean = float(np.mean([np.min(np.hypot(*(boundary - p).T)) for p in placed]))
    robust = _robust_corner_residual(placed, boundary)     # keep=0.8 -> drops the worst corner
    assert mean > 2.0
    assert robust < 0.6, f"robust should discard the single outlier corner, got {robust:.2f}"
    assert robust < mean                        # strictly the point: robust < brittle mean


def test_robust_never_hides_a_globally_bad_fit():
    """If MANY corners are off (a genuinely wrong placement), trimming one does NOT rescue it."""
    boundary, coords = _square_boundary(side=40.0)
    placed = np.array(coords, float) + np.array([25.0, 25.0])   # whole ring shifted 25 m off
    robust = _robust_corner_residual(placed, boundary)
    assert robust > 10.0, "a wholesale misplacement must remain a large residual"


def _fit_stub(mean_resid, robust_resid):
    return types.SimpleNamespace(rot_residual=mean_resid, rot_residual_robust=robust_resid)


def test_gate_residual_switch_off_uses_plain_mean(monkeypatch):
    monkeypatch.setattr(cadastral_seat, "CAD_ROBUST_RESID", False)
    fit = _fit_stub(18.0, 6.0)
    assert cadastral_seat._gate_residual(fit) == 18.0     # shipped behaviour unchanged


def test_gate_residual_switch_on_takes_the_lower(monkeypatch):
    monkeypatch.setattr(cadastral_seat, "CAD_ROBUST_RESID", True)
    fit = _fit_stub(18.0, 6.0)
    assert cadastral_seat._gate_residual(fit) == 6.0      # recovers the jittered-corner fit


def test_gate_residual_switch_on_never_raises_residual(monkeypatch):
    """min() can only lower the residual -> the switch is strictly recall-additive, never
    stricter (a fit already passing on the mean can never be pushed to fail)."""
    monkeypatch.setattr(cadastral_seat, "CAD_ROBUST_RESID", True)
    fit = _fit_stub(4.0, 9.0)                              # robust happens to be higher
    assert cadastral_seat._gate_residual(fit) == 4.0      # still the mean; never worse


def _m1_from_ring(ring):
    stones = [M1Stone(x=float(x), y=float(y), label=str(i), index=i)
              for i, (x, y) in enumerate(ring)]
    m1 = M1PlotData(stones=stones, survey_number="TEST")
    m1.outer_stone_indices = list(range(len(ring)))
    return m1


def test_fit_populates_robust_residual_field():
    """A real rigid fit fills rot_residual_robust; on a clean fit it tracks rot_residual."""
    plot = np.array([(0.0, 0.0), (60.0, 0.0), (60.0, 20.0), (0.0, 20.0)])
    t = np.array([700000.0, 1200000.0])
    parcel = CadastralParcel(survey_number="TEST",
                             polygon=Polygon([tuple(p + t) for p in plot]),
                             village="V", source_crs="EPSG:32643")
    fit = fit_plot_to_parcel(_m1_from_ring([tuple(p) for p in plot]), parcel)
    assert fit is not None and fit.method == "rigid"
    assert np.isfinite(fit.rot_residual_robust)
    assert fit.rot_residual_robust <= fit.rot_residual + 1e-6   # trim can only lower it
