"""stone_verify: MATCH when corners sit on true stones; SHIFTED when shape ok but placed off."""
from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_club.stone_verify import verify_stones


def _square(x0, y0, side=100.0):
    return np.array([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side),
                     (x0 + side / 2, y0)], float)


def test_match_when_corners_on_true_stones():
    truth = _square(0, 0)
    rep = verify_stones({"A": _square(0, 0)}, truth, tol=2.5)
    r = rep.rows[0]
    assert r.verdict == "MATCH"
    assert r.median_err < 0.5
    assert rep.n_match == 1


def test_shifted_when_shape_ok_but_placed_off():
    # true stones form a square; our FMB is the SAME square translated 15 m -> shape
    # matches (rigid fit is exact) but every corner is ~15 m from a true stone.
    truth = _square(0, 0)
    shifted = _square(0, 0) + np.array([15.0, 0.0])
    rep = verify_stones({"A": shifted}, truth, tol=2.5)
    r = rep.rows[0]
    assert r.median_err > 3.0
    assert r.congruent is True
    assert r.verdict == "SHIFTED"
    assert rep.n_shape_ok == 1


def test_shape_check_when_no_rigid_fit():
    # our "FMB" is random noise -> no rigid alignment to the true square.
    rng = np.random.default_rng(0)
    truth = _square(0, 0)
    noise = truth.mean(0) + rng.normal(0, 40, size=(6, 2))
    rep = verify_stones({"A": noise}, truth, tol=2.5)
    assert rep.rows[0].verdict == "SHAPE_CHECK"


def test_empty_truth_is_safe():
    rep = verify_stones({"A": _square(0, 0)}, np.empty((0, 2)))
    assert rep.rows == [] and rep.n_truth_stones == 0
