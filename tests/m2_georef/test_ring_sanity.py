"""Ring sanity guard in geometric_match -- M1-quality observability (Rule #3).

A duplicate corner vertex is an M1 extraction defect that must surface in the log
(fixed upstream in M1, never by loosening M2/M3 gates). Matching refuses only the
truly degenerate ring (< 3 DISTINCT corners), where 2D congruence is undefined.
Synthetic geometry only.
"""
from __future__ import annotations

import logging

from landintel.pipeline.m2_georef.extract_m1 import M1PlotData, M1Stone
from landintel.pipeline.m2_georef.extract_surveyor import SurveyorData, SurveyorStone
from landintel.pipeline.m2_georef.match import geometric_match


def _m1(points, corners):
    stones = [M1Stone(x=float(x), y=float(y), label=f"S{i}", index=i)
              for i, (x, y) in enumerate(points)]
    return M1PlotData(stones=stones, survey_number="T",
                      source_file="synthetic.dxf",
                      outer_stone_indices=list(corners))


def _surveyor(points):
    sd = SurveyorData(source_file="synthetic_surveyor.dxf")
    sd.stones = [SurveyorStone(x=float(x), y=float(y), code="B", index=i)
                 for i, (x, y) in enumerate(points)]
    sd.build_index()
    return sd


def test_duplicate_corner_logs_warning_but_still_tries(caplog):
    # 4 corners, two identical -> warn (M1 fidelity), but 3 distinct remain so
    # matching itself still runs.
    pts = [(0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (40.0, 30.0)]
    m1 = _m1(pts, [0, 1, 2, 3])
    surv = _surveyor([(1000.0, 2000.0), (1040.0, 2000.0), (1040.0, 2030.0)])
    with caplog.at_level(logging.WARNING, logger="landintel.pipeline.m2_georef.match"):
        geometric_match(m1, surv)
    assert any("duplicate vertices" in r.message and "M1" in r.message
               for r in caplog.records)


def test_fewer_than_three_distinct_corners_is_no_match(caplog):
    # 3 corners but only 2 distinct -> degenerate, honestly refused (no fabricated fit).
    pts = [(0.0, 0.0), (40.0, 0.0), (40.0, 0.0)]
    m1 = _m1(pts, [0, 1, 2])
    surv = _surveyor([(0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (0.0, 30.0)])
    with caplog.at_level(logging.WARNING, logger="landintel.pipeline.m2_georef.match"):
        res = geometric_match(m1, surv)
    assert not res.matched
    assert any("degenerate corner ring" in r.message for r in caplog.records)


def test_clean_ring_no_warning(caplog):
    pts = [(0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (0.0, 30.0)]
    m1 = _m1(pts, [0, 1, 2, 3])
    surv = _surveyor([(500.0, 500.0), (540.0, 500.0), (540.0, 530.0), (500.0, 530.0)])
    with caplog.at_level(logging.WARNING, logger="landintel.pipeline.m2_georef.match"):
        geometric_match(m1, surv)
    assert not any("duplicate vertices" in r.message for r in caplog.records)
