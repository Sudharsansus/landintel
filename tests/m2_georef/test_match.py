"""Unit tests for fingerprint + neighborhood matching."""

from __future__ import annotations

from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf
from landintel.pipeline.m2_georef.extract_surveyor import extract_surveyor
from landintel.pipeline.m2_georef.match import (
    _bin,
    _fingerprint,
    _fingerprint_rms,
    geometric_match,
    match_plot,
)


def test_bin_rounds_to_half_metre():
    assert _bin(44.2) == 44.0
    assert _bin(44.3) == 44.5
    assert _bin(51.66) == 51.5


def test_fingerprint_is_sorted_and_binned():
    n, fp = _fingerprint([51.66, 22.0, 50.0, 35.0])
    assert n == 4
    assert fp == (22.0, 35.0, 50.0, 51.5)  # sorted, binned


def test_fingerprint_rms_identical_is_zero():
    fp = _fingerprint([10.0, 20.0, 30.0])
    assert _fingerprint_rms(fp, fp) == 0.0


def test_fingerprint_rms_mismatched_counts_is_inf():
    fa = _fingerprint([10.0, 20.0])
    fb = _fingerprint([10.0, 20.0, 30.0])
    assert _fingerprint_rms(fa, fb) == float("inf")


def test_match_plot_finds_correct_window(m1_dxf, surveyor_dxf):
    """The 4-corner plot must match its 4-stone surveyor target, not the decoy."""
    m1 = extract_m1_dxf(m1_dxf)
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()

    result = match_plot(m1, surveyor)

    assert result.matched, f"expected a match, got {result.match_method}"
    # Exact synthetic data -> near-zero residual.
    assert result.fingerprint_score < 1.0
    # All four M1 corners mapped onto distinct surveyor stones.
    mapped = [j for j in result.stone_map if j >= 0]
    assert len(mapped) == 4
    assert len(set(mapped)) == 4
    # The matched surveyor stones are the target plot (indices 0..3), not decoy.
    assert set(mapped) == {0, 1, 2, 3}


def test_geometric_match_finds_congruent_template(m1_dxf, surveyor_dxf):
    """Geometric congruence locates the plot's exact shape in the stone cloud."""
    m1 = extract_m1_dxf(m1_dxf)
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()

    g = geometric_match(m1, surveyor)

    assert g.matched
    assert g.n_matched_stones == 4          # all four corners are inliers
    assert g.fingerprint_score < 1.0        # sub-metre inlier residual (exact)
    assert set(j for j in g.stone_map if j >= 0) == {0, 1, 2, 3}


def test_geometric_match_rejects_noncongruent(surveyor_dxf, tmp_path):
    """A plot whose shape is absent from the surveyor cloud is NOT matched
    (the core false-positive guard)."""
    from landintel.core.models import Boundary, CornerPoint, Plot
    from landintel.pipeline.m1_extract.to_dxf import write_dxf

    # A skinny quadrilateral unlike the synthetic target or its decoy.
    verts = [(0.0, 0.0), (3.0, 0.0), (3.3, 41.0), (0.2, 40.0)]
    plot = Plot(
        client_id="c", survey_no="555", district="E", taluk="P", village="INGUR",
        scale=1000, stated_area=0.012,
        boundary=Boundary(points=verts + [verts[0]]),
        corner_points=[CornerPoint(label=str(i), x=x, y=y)
                       for i, (x, y) in enumerate(verts)],
    )
    odd = write_dxf(plot, tmp_path / "m1_555.dxf")
    m1 = extract_m1_dxf(odd)
    surveyor = extract_surveyor(surveyor_dxf)
    surveyor.build_index()

    g = geometric_match(m1, surveyor)
    assert not g.matched
