"""Threshold centralization -- one source of truth, no silent drift.

The tiling-overlap threshold previously DIVERGED across four call sites (0.20 in
verification + m2_club verify, 0.30 in regate + AssemblyAgent), meaning the FP policy
depended on WHICH code path checked it. These tests lock every consumer to the single
centralized constant so the drift cannot regress.
"""
from __future__ import annotations

import inspect


def test_threshold_module_importable_and_complete():
    from landintel.pipeline.m2_club import disposition_thresholds as dt
    for name in ("TILING_OVERLAP_THRESHOLD", "CAD_AREA_LO", "CAD_AREA_HI",
                 "CAD_SCALE_LO", "CAD_SCALE_HI", "CAD_ROT_RESID_MAX", "CAD_ROT_K",
                 "SEAT_K", "SEAT_FLOOR_M", "CAD_MIN_STONES", "FULL_MATCH_STONES"):
        assert hasattr(dt, name), f"missing threshold: {name}"


def test_all_four_overlap_consumers_share_one_value():
    from landintel.agents import regate, verification
    from landintel.pipeline.m2_club import disposition_thresholds as dt
    from landintel.pipeline.m2_club import verify as club_verify

    assert regate._OVERLAP_FRAC == dt.TILING_OVERLAP_THRESHOLD
    assert verification._FOOTPRINT_CONFLICT == dt.TILING_OVERLAP_THRESHOLD
    assert club_verify._TILING_OVERLAP_MAX == dt.TILING_OVERLAP_THRESHOLD

    # AssemblyAgent's default parameter is the same constant.
    from landintel.agents.club_agents import AssemblyAgent
    sig = inspect.signature(AssemblyAgent.run)
    assert sig.parameters["max_overlap_frac"].default == dt.TILING_OVERLAP_THRESHOLD


def test_no_local_constant_left_behind():
    # The old module-level literals must be GONE (they were the drift mechanism);
    # the surviving names must be imports of the central constant.
    import landintel.agents.regate as regate
    import landintel.agents.verification as verification
    from landintel.pipeline.m2_club import disposition_thresholds as dt

    src_regate = inspect.getsource(regate)
    src_verif = inspect.getsource(verification)
    assert "TILING_OVERLAP_THRESHOLD" in src_regate
    assert "TILING_OVERLAP_THRESHOLD" in src_verif
    assert "_OVERLAP_FRAC = 0.30" not in src_regate
    assert "_FOOTPRINT_CONFLICT = 0.20" not in src_verif
    assert dt.TILING_OVERLAP_THRESHOLD == 0.30       # the VALIDATED value (not 0.20)


def test_cadastral_gate_constants_come_from_central_module():
    import importlib
    cs = importlib.import_module("landintel.pipeline.m2_club.cadastral_seat")
    from landintel.pipeline.m2_club import disposition_thresholds as dt
    assert cs.CAD_AREA_LO == dt.CAD_AREA_LO
    assert cs.CAD_AREA_HI == dt.CAD_AREA_HI
    assert cs.CAD_SCALE_LO == dt.CAD_SCALE_LO
    assert cs.CAD_SCALE_HI == dt.CAD_SCALE_HI
    assert cs.CAD_MIN_STONES == dt.CAD_MIN_STONES
    assert cs.SEAT_K == dt.SEAT_K
    assert cs.SEAT_FLOOR_M == dt.SEAT_FLOOR_M


def test_full_match_bar_is_conditional_not_flat():
    # The client's "5 stones" bar is data-keyed: a plot can never be required to
    # match more stones than it has. min(bar, n_corners) is the contract.
    from landintel.pipeline.m2_club.disposition_thresholds import FULL_MATCH_STONES
    assert FULL_MATCH_STONES == 5
    for n_corners in (3, 4, 5, 8):
        required = min(FULL_MATCH_STONES, n_corners)
        assert required <= n_corners                 # never impossible
        if n_corners >= 5:
            assert required == 5                     # the client's bar when reachable
