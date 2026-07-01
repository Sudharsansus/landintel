"""Unit tests for boundary/generic vertex warping."""

from __future__ import annotations

import numpy as np
import pytest

from landintel.pipeline.m2_georef.warp import (
    warp_boundary_vertices,
    warp_generic_vertices,
    warp_point,
)


def test_warp_point_midpoint():
    """A point at fraction t along the original edge maps to t along the adjusted."""
    pt = (5.0, 0.0)  # midpoint of (0,0)-(10,0)
    out = warp_point(pt, (0.0, 0.0), (10.0, 0.0), (100.0, 100.0), (100.0, 200.0))
    assert out == pytest.approx([100.0, 150.0])


def test_warp_point_endpoints():
    out_a = warp_point((0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (5.0, 5.0), (25.0, 5.0))
    out_b = warp_point((10.0, 0.0), (0.0, 0.0), (10.0, 0.0), (5.0, 5.0), (25.0, 5.0))
    assert out_a == pytest.approx([5.0, 5.0])
    assert out_b == pytest.approx([25.0, 5.0])


def test_warp_point_degenerate_edge():
    """Zero-length original edge falls back to the adjusted midpoint."""
    out = warp_point((1.0, 1.0), (2.0, 2.0), (2.0, 2.0), (10.0, 0.0), (20.0, 0.0))
    assert out == pytest.approx([15.0, 0.0])


def test_warp_boundary_vertices_translation():
    """Stones moved by a constant offset carry intermediate vertices along."""
    verts = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]  # middle is a non-stone vertex
    stone_idx = [0, -1, 1]
    orig = np.array([[0.0, 0.0], [10.0, 0.0]])
    adj = orig + np.array([100.0, 50.0])
    out = warp_boundary_vertices(verts, stone_idx, orig, adj)
    assert out[0] == pytest.approx([100.0, 50.0])
    assert out[1] == pytest.approx([105.0, 50.0])  # midpoint follows
    assert out[2] == pytest.approx([110.0, 50.0])


def test_warp_generic_vertices_uniform_offset():
    """When every stone shares one offset, IDW reproduces that offset exactly."""
    orig = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    offset = np.array([500.0, -300.0])
    adj = orig + offset
    verts = [(5.0, 5.0), (2.0, 8.0), (9.0, 1.0)]
    out = warp_generic_vertices(verts, orig, adj)
    for v, o in zip(verts, out):
        assert o == pytest.approx(np.array(v) + offset, abs=1e-6)


def test_warp_generic_vertices_single_stone():
    """Robust when only one stone exists (k_neighbors capped to available)."""
    orig = np.array([[0.0, 0.0]])
    adj = np.array([[100.0, 100.0]])
    out = warp_generic_vertices([(1.0, 1.0)], orig, adj)
    assert out[0] == pytest.approx([101.0, 101.0])
