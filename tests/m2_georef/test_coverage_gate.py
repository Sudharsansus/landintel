"""Regression tests for the chain-coverage false-positive gate + the global
footprint non-overlap resolver.

These guard the fix for the dense-cloud OVER-MATCH problem: per-plot geometric
congruence finds *some* congruent stone subset for nearly every plot in a dense
surveyor cloud (35/35 matched, 92 footprint overlaps on real INGUR data). The
decisive discriminator is CHAIN COVERAGE -- only a truly-surveyed plot's boundary
lies on the surveyor's traced SITE DATA LINE -- backstopped by a global rule that
the ACCEPT set must be a non-overlapping tiling.
"""

from __future__ import annotations

import ezdxf

from landintel.core.enums import LayerType
from landintel.pipeline.m2_georef.pipeline import (
    GeorefResult,
    _footprint_polygon,
    _resolve_footprint_conflicts,
)
from landintel.pipeline.m2_georef.verify import chain_coverage


# --- chain_coverage --------------------------------------------------------


def _square_segments(x0, y0, s):
    pts = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s), (x0, y0)]
    return [(pts[i], pts[i + 1]) for i in range(4)]


def test_chain_coverage_full_when_boundary_on_traced_line():
    """A boundary lying exactly on a traced polyline -> coverage ~1.0."""
    seg = _square_segments(0, 0, 100)
    traced = [[(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]]  # same ring
    assert chain_coverage(seg, traced, tol=3.0) > 0.99


def test_chain_coverage_zero_when_boundary_far_from_traced():
    """A boundary nowhere near any traced line -> coverage 0."""
    seg = _square_segments(0, 0, 100)
    traced = [[(5000, 5000), (5100, 5000), (5100, 5100)]]  # far away
    assert chain_coverage(seg, traced, tol=3.0) == 0.0


def test_chain_coverage_partial_when_half_traced():
    """Only some edges traced -> coverage strictly between 0 and 1."""
    seg = _square_segments(0, 0, 100)
    # Trace only the bottom and right edges (2 of 4 -> ~50% of perimeter).
    traced = [[(0, 0), (100, 0), (100, 100)]]
    cov = chain_coverage(seg, traced, tol=3.0)
    assert 0.4 < cov < 0.6


def test_chain_coverage_empty_inputs():
    assert chain_coverage([], [[(0, 0), (1, 1)]]) == 0.0
    assert chain_coverage(_square_segments(0, 0, 10), []) == 0.0


# --- global footprint non-overlap resolver ---------------------------------


def _write_boundary_dxf(path, x0, y0, s):
    """Write a minimal georef DXF whose BOUNDARY layer is one closed square."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add(LayerType.BOUNDARY.value)
    pts = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s), (x0, y0)]
    msp.add_lwpolyline(pts, dxfattribs={"layer": LayerType.BOUNDARY.value})
    doc.saveas(path)
    return str(path)


def _accept(survey, path, cov):
    return GeorefResult(
        m1_file=path, survey_number=survey, matched=True, output_file=path,
        n_corners=8, n_inliers=8, chain_coverage=cov, recommendation="ACCEPT",
    )


def test_resolver_demotes_lower_coverage_of_overlapping_accepts(tmp_path):
    """Two ACCEPT plots whose footprints overlap -> keep higher coverage, demote
    the other to REVIEW (real parcels tile; they cannot occupy the same ground)."""
    a = _write_boundary_dxf(tmp_path / "a.dxf", 0, 0, 100)        # 0..100
    b = _write_boundary_dxf(tmp_path / "b.dxf", 20, 20, 100)      # heavy overlap
    ra = _accept("A", a, cov=0.90)   # higher coverage -> kept
    rb = _accept("B", b, cov=0.60)   # lower coverage  -> demoted
    _resolve_footprint_conflicts([ra, rb])
    assert ra.recommendation == "ACCEPT"
    assert rb.recommendation == "REVIEW"
    assert "footprint overlaps" in rb.error


def test_resolver_keeps_both_when_only_edge_sharing(tmp_path):
    """Adjacent non-overlapping plots (shared edge, disjoint interiors) both stay
    ACCEPT -- the tiling case must not be demoted."""
    a = _write_boundary_dxf(tmp_path / "a2.dxf", 0, 0, 100)       # 0..100
    b = _write_boundary_dxf(tmp_path / "b2.dxf", 100, 0, 100)     # 100..200, touches edge
    ra = _accept("A", a, cov=0.90)
    rb = _accept("B", b, cov=0.85)
    _resolve_footprint_conflicts([ra, rb])
    assert ra.recommendation == "ACCEPT"
    assert rb.recommendation == "ACCEPT"


def test_footprint_polygon_reads_boundary(tmp_path):
    path = _write_boundary_dxf(tmp_path / "fp.dxf", 10, 10, 50)
    poly = _footprint_polygon(path)
    assert poly is not None
    assert poly.area == 2500.0  # 50 x 50
