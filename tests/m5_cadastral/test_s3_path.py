"""Unit tests for the S3 cadastral path: fuzzy label matching, rigid fit
(scale~1 + shape preservation), and the A2 orientation/flip gate.

These exercise the pure logic the live S3 run depends on, with no network/tiles.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m2_georef"))

from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf
from landintel.pipeline.m5_cadastral.fit import (
    _is_near_square,
    _orientation_consistent,
    _principal_angle,
    fit_plot_to_parcel,
)
from landintel.pipeline.m5_cadastral.s3_tiles import _fuzzy_survey_match
from landintel.pipeline.m5_cadastral.source import CadastralParcel
from shapely.geometry import Polygon


# ---------------------------------------------------------------- fuzzy OCR ----
def test_fuzzy_exact_and_punctuation():
    known = {"1019", "077", "668"}
    assert _fuzzy_survey_match("1019", known) == "1019"
    assert _fuzzy_survey_match("10/19", known) == "1019"   # slash artifact
    assert _fuzzy_survey_match("10 19", known) == "1019"   # split read
    assert _fuzzy_survey_match("O77", known) == "077"      # O->0


def test_fuzzy_closed_set_never_invents():
    # A read with no member of the known set returns None (cannot invent a survey).
    assert _fuzzy_survey_match("4242", {"1019", "668"}) is None
    assert _fuzzy_survey_match("garbage", {"668"}) is None


def test_fuzzy_prefers_longest_candidate():
    # When both a short and long survey could match, the longer wins so a 2-digit
    # subdivision label can't shadow the real 4-digit survey.
    assert _fuzzy_survey_match("10 19", {"10", "1019"}) == "1019"


# ----------------------------------------------------------- near-square -------
def test_is_near_square():
    square = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    elong = np.array([(0.0, 0.0), (50.0, 0.0), (50.0, 8.0), (0.0, 8.0)])
    assert _is_near_square(square)
    assert not _is_near_square(elong)


# ---------------------------------------------------------- flip detection -----
def _parcel(poly_coords, sn="784"):
    return CadastralParcel(survey_number=sn, polygon=Polygon(poly_coords),
                           village="INGUR", source_crs="EPSG:32643")


def _rot(deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def test_orientation_accepts_aligned():
    ring = np.array([(0.0, 0.0), (50.0, 0.0), (50.0, 10.0), (0.0, 10.0)])
    parcel = _parcel([(0, 0), (50, 0), (50, 10), (0, 10)])
    assert _orientation_consistent(np.eye(2), parcel, ring)


def test_orientation_rejects_90deg_flip():
    ring = np.array([(0.0, 0.0), (50.0, 0.0), (50.0, 10.0), (0.0, 10.0)])
    parcel = _parcel([(0, 0), (50, 0), (50, 10), (0, 10)])
    # A 90-degree rotation puts the long axis across the parcel's long axis.
    assert not _orientation_consistent(_rot(90.0), parcel, ring)


def test_orientation_skips_near_square():
    ring = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    parcel = _parcel([(0, 0), (10, 0), (10, 10), (0, 10)])
    # No stable axis -> the gate passes (carried by area/scale gates instead).
    assert _orientation_consistent(_rot(90.0), parcel, ring)


# ------------------------------------------------- rigid fit: scale + shape ----
def test_rigid_fit_scale_near_one_and_shape_preserved(m1_dxf):
    from conftest import PLOT_VERTS, apply_true_transform
    utm = apply_true_transform(np.array(PLOT_VERTS))
    parcel = _parcel([tuple(p) for p in utm])

    m1 = extract_m1_dxf(m1_dxf)
    fit = fit_plot_to_parcel(m1, parcel)
    assert fit is not None
    assert fit.method == "rigid"
    # M1 is metres, parcel is metres -> a correct placement aligns at scale ~1.
    assert 0.9 < fit.s < 1.1
    assert fit.orientation_ok

    # Shape preserved: edge lengths of the placed corner ring equal the M1 ring's.
    corners = m1.outer_stone_indices
    src = m1.stone_positions()[np.array(corners)]
    placed = fit.adjusted[np.array(corners)]
    for i in range(len(corners)):
        a, b = (i, (i + 1) % len(corners))
        d_src = float(np.hypot(*(src[b] - src[a])))
        d_pl = float(np.hypot(*(placed[b] - placed[a])))
        assert abs(d_src - d_pl) < 1e-3, "rigid transform must preserve edge lengths"


def test_principal_angle_axis_aligned():
    coords = np.array([(0.0, 0.0), (50.0, 0.0), (50.0, 5.0), (0.0, 5.0)])
    # Long axis along x -> principal angle ~0 (mod pi).
    ang = _principal_angle(coords) % math.pi
    assert ang < 0.1 or abs(ang - math.pi) < 0.1
