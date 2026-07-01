"""Tests for boundary-edge registration onto the surveyor's traced lines.

The decisive capability: place a plot by fitting its boundary EDGES onto the
SITE DATA LINE (the actually-traced cadastral boundary) and scoring by chain
coverage -- so a corridor-clipped plot that exposes a traceable edge but too few
corner stones can still be georeferenced. The accept decision is the caller's
(coverage bar + identity + window + non-overlap); these tests pin the mechanics.
"""

from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf
from landintel.pipeline.m2_georef.extract_surveyor import extract_surveyor
from landintel.pipeline.m2_georef.edge_register import (
    register_plot_on_traced,
    traced_segments,
)


def test_register_places_traced_plot(m1_dxf, surveyor_dxf):
    """A plot whose boundary IS traced registers at high coverage, at its true
    UTM seat (the synthetic surveyor traces the full ring)."""
    from conftest import PLOT_VERTS, apply_true_transform

    m1 = extract_m1_dxf(m1_dxf)
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()

    reg = register_plot_on_traced(m1, surveyor)
    assert reg is not None, "should find a registration on the traced ring"
    assert reg.coverage >= 0.5, f"traced plot should cover well (got {reg.coverage:.2f})"

    # The placed corner ring should land on the known UTM positions.
    expected = apply_true_transform(np.array(PLOT_VERTS))
    placed = reg.adjusted[[s.index for s in m1.stones]][:4]
    # Match each expected corner to its nearest placed corner (order-independent).
    for e in expected:
        d = np.min(np.linalg.norm(placed - e, axis=1))
        assert d < 3.0, f"placed corner off by {d:.2f} m from true UTM"


def test_traced_segments_window_filter(surveyor_dxf):
    """A stone window restricts the traced segments to that neighbourhood."""
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    n = len(surveyor.stones)

    all_segs = traced_segments(surveyor)
    assert len(all_segs) >= 4, "the full ring trace yields >=4 segments"

    # Window over only the 4 target stones (indices 0..3) -> excludes the decoy.
    mask = np.zeros(n, dtype=bool)
    mask[:4] = True
    win_segs = traced_segments(surveyor, keep=mask)
    assert win_segs, "target window keeps the plot's traced segments"
    assert len(win_segs) <= len(all_segs)


def test_register_none_without_ring(m1_dxf, surveyor_dxf):
    """No usable corridor segments -> returns None (never a bogus placement)."""
    m1 = extract_m1_dxf(m1_dxf)
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()
    reg = register_plot_on_traced(m1, surveyor, segments=[])
    assert reg is None
