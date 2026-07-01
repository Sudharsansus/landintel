"""All-angle rotation robustness -- the core M2 math must place a STRAIGHT (axis-aligned) FMB
at the surveyor's TRUE real-world orientation, for ANY rotation, and absorb few-metre field
noise without breaking. These tests pin the Umeyama similarity transform that does it.

Why this matters: M1 draws every plot straight; the surveyor raw-data DXF has it rotated to
its actual bearing. If rotation recovery fails at any angle (or silently reflects a symmetric
plot), the georeferenced map is wrong. So we sweep 0-360 deg and check proper-rotation + noise.
"""
from __future__ import annotations

import glob
import math

import numpy as np
import pytest

from landintel.pipeline.m2_georef.transform import umeyama


def _R(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.atan2(R[1, 0], R[0, 0])) % 360.0


def _circular_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# asymmetric 5-corner plot (no rotational symmetry -> unambiguous rotation)
_PLOT = np.array([[0, 0], [120, 0], [120, 80], [55, 110], [0, 60]], float)


def test_umeyama_recovers_every_rotation_angle():
    t_true = np.array([783000.0, 1241000.0])          # UTM-scale translation
    for deg in range(0, 360, 10):
        th = math.radians(deg)
        dst = (_PLOT @ _R(th).T) + t_true             # scale 1 (M1 already metric)
        R, s, t, res = umeyama(_PLOT, dst)
        assert _circular_diff(_angle_deg(R), deg) < 1e-3, (deg, _angle_deg(R))
        assert abs(s - 1.0) < 1e-6
        assert np.linalg.det(R) > 0                    # PROPER rotation, never reflected
        assert res.max() < 1e-6                        # exact placement when noise-free


def test_umeyama_absorbs_few_metre_field_noise():
    rng = np.random.default_rng(7)
    th = math.radians(37.0)
    dst = (_PLOT @ _R(th).T) + np.array([783000.0, 1241000.0])
    noisy = dst + rng.normal(0.0, 1.5, dst.shape)     # ~1.5 m per-coord field noise
    R, s, t, res = umeyama(_PLOT, noisy)
    assert _circular_diff(_angle_deg(R), 37.0) < 3.0  # rotation still recovered within a few deg
    assert abs(s - 1.0) < 0.05                         # scale stays ~1 (no spurious stretch)
    assert res.mean() < 4.0                            # noise absorbed into residual, not blown up


def test_umeyama_no_reflection_on_symmetric_square():
    # a perfectly symmetric square is the reflection trap: must still be a PROPER rotation
    sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    for deg in (0, 45, 90, 123, 270):
        th = math.radians(deg)
        dst = (sq @ _R(th).T) + np.array([5.0, 9.0])
        R, s, t, res = umeyama(sq, dst)
        assert np.linalg.det(R) > 0, deg               # no improper/flipped solution
        assert res.max() < 1e-6


# ---- matcher-level: rotate a REAL FMB through all angles, verify geometric_match recovers it --
def _load_real_m1():
    from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf
    for p in sorted(glob.glob("test2/INGUR/m1_outputs/*.dxf")):
        try:
            m1 = extract_m1_dxf(p)
            if len(m1.outer_stone_indices) >= 4:
                return m1
        except Exception:  # noqa: BLE001
            continue
    return None


def test_geometric_match_recovers_real_fmb_at_every_angle():
    """A STRAIGHT M1 plot, rotated to an arbitrary bearing and dropped into a synthetic
    surveyor cloud, must be found by geometric_match at its correct angle for ALL rotations."""
    from landintel.pipeline.m2_georef.extract_surveyor import SurveyorData, SurveyorStone
    from landintel.pipeline.m2_georef.match import geometric_match
    m1 = _load_real_m1()
    if m1 is None:
        pytest.skip("no INGUR M1 outputs on disk")
    pos = m1.stone_positions()
    n_corners = len(m1.outer_stone_indices)
    base = np.array([783000.0, 1241000.0])
    n_ok = 0
    for deg in range(0, 360, 30):
        th = math.radians(deg)
        rot = (pos @ _R(th).T) + base                  # rotate the whole plot to a real bearing
        surv = SurveyorData(stones=[SurveyorStone(float(x), float(y), "B", i)
                                    for i, (x, y) in enumerate(rot)])
        mr = geometric_match(m1, surv)
        assert mr.matched, f"no match at {deg} deg"
        assert mr.n_matched_stones >= n_corners - 1, (deg, mr.n_matched_stones, n_corners)
        # recover the angle from the matched correspondences and check it == deg
        pairs = [(i, j) for i, j in enumerate(mr.stone_map) if j >= 0]
        src = np.array([pos[i] for i, _ in pairs])
        dst = np.array([rot[j] for _, j in pairs])
        R, s, t, res = umeyama(src, dst)
        assert _circular_diff(_angle_deg(R), deg) < 1.0, (deg, _angle_deg(R))
        n_ok += 1
    assert n_ok == 12
