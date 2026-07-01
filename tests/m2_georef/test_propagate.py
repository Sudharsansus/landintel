"""Tests for the second-pass corridor propagation building blocks.

The decisive new capability is ``geometric_match(allowed_stones=...)``: matching
restricted to a corridor WINDOW so a near-congruent plot cannot grab a foreign
seat. The propagation pass relies on it to re-place REVIEW plots in their correct
schedule neighbourhood, and is a strict no-op when it lacks anchors.
"""

from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf
from landintel.pipeline.m2_georef.extract_surveyor import extract_surveyor
from landintel.pipeline.m2_georef.match import geometric_match
from landintel.pipeline.m2_georef.pipeline import seed_place
from landintel.pipeline.m2_georef.propagate import propagate_review_plots


def test_allowed_stones_window_restricts_match(m1_dxf, surveyor_dxf):
    """A window that INCLUDES the plot's true stones matches; a window that
    EXCLUDES them does not (the matcher cannot reach a foreign seat)."""
    m1 = extract_m1_dxf(m1_dxf)
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    n = len(surveyor.stones)

    # conftest builds the 4 target stones FIRST (indices 0..3), decoy after.
    target_mask = np.zeros(n, dtype=bool)
    target_mask[:4] = True
    in_window = geometric_match(m1, surveyor, allowed_stones=target_mask)
    assert in_window.matched, "should match when its true stones are in the window"

    # Exclude the 4 target stones -> only the (non-congruent) decoy remains.
    no_target = ~target_mask
    out_window = geometric_match(m1, surveyor, allowed_stones=no_target)
    assert not out_window.matched, "must NOT match outside its true window"


def test_unrestricted_match_still_works(m1_dxf, surveyor_dxf):
    """allowed_stones=None preserves the original full-cloud behaviour."""
    m1 = extract_m1_dxf(m1_dxf)
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    assert geometric_match(m1, surveyor).matched


def test_seed_place_two_points(m1_dxf, surveyor_dxf, tmp_path):
    """Two operator-given corner->UTM correspondences place the whole plot
    (Stage-3 automated): ACCEPT_SEEDED, sub-metre residual, output written."""
    import numpy as np
    from conftest import PLOT_VERTS, STONE_LABELS, apply_true_transform

    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    utm = apply_true_transform(np.array(PLOT_VERTS))
    a = STONE_LABELS.index("1")
    b = STONE_LABELS.index("3")
    r = seed_place(m1_dxf, surveyor, "1", "3",
                   tuple(utm[a]), tuple(utm[b]), tmp_path / "seed")
    assert r.recommendation == "ACCEPT_SEEDED", r.error
    assert r.output_file
    assert r.field_residual_max < 1.0
    assert r.verify_result and r.verify_result.all_passed


def test_seed_place_unknown_corner_label_is_review(m1_dxf, surveyor_dxf, tmp_path):
    """An unknown corner label is surfaced as REVIEW with a reason, not a crash."""
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    r = seed_place(m1_dxf, surveyor, "ZZ", "3",
                   (783000.0, 1241000.0), (783050.0, 1241050.0), tmp_path / "s2")
    assert r.recommendation == "REVIEW"
    assert "corner label" in r.error


def test_propagation_noop_without_anchors(m1_dxf, surveyor_dxf, tmp_path):
    """With no ACCEPT anchors, propagation places nothing (needs >=2 anchors)."""
    from landintel.pipeline.m2_georef.pipeline import GeorefResult
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    m1 = extract_m1_dxf(m1_dxf)
    # One REVIEW plot, zero anchors.
    r = GeorefResult(m1_file=str(m1_dxf), survey_number="784", matched=False,
                     recommendation="REVIEW")
    upgraded = propagate_review_plots(
        surveyor, [r], {"784": m1}, ["784"], tmp_path)
    assert upgraded == []
    assert r.recommendation == "REVIEW"   # untouched
