"""Extraction tests: M1 DXF round-trip and the real INGUR surveyor DXF.

The M1 extraction test closes the loop with M1: a plot serialized by the real
``to_dxf`` must be parsed back by ``extract_m1_dxf`` with stones, edges, and the
outer boundary intact -- proving the layer-name binding to ``LayerType`` tracks
M1's actual output (including ``"neighbor label"``, lowercase with a space).
"""

from __future__ import annotations

import pytest

from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf
from landintel.pipeline.m2_georef.extract_surveyor import (
    BOUNDARY_CODES,
    extract_surveyor,
)

from conftest import REAL_SURVEYOR_DXF


def test_extract_m1_roundtrip(m1_dxf):
    """Parse the synthetic M1 DXF back; stones/edges/outer-boundary intact."""
    m1 = extract_m1_dxf(m1_dxf)

    assert m1.survey_number == "784"
    assert m1.n_stones == 4
    assert {s.label for s in m1.stones} == {"1", "2", "3", "4"}

    # Four boundary edges, perimeter ~158.66 m.
    assert m1.n_edges == 4
    assert m1.total_perimeter == pytest.approx(158.66, abs=0.5)

    # Outer-boundary cycle recovered with all four stones.
    m1.extract_outer_boundary()
    assert len(m1.outer_stone_indices) == 4
    assert len(m1.outer_edges) == 4
    assert m1.outer_perimeter == pytest.approx(158.66, abs=0.5)


def test_extract_m1_edge_lengths(m1_dxf):
    """Edge-length fingerprint matches the known plot edges."""
    m1 = extract_m1_dxf(m1_dxf)
    lengths = sorted(e.length_m for e in m1.outer_edges)
    expected = sorted([50.0, 35.0, 51.662, 22.0])
    for a, b in zip(lengths, expected):
        assert a == pytest.approx(b, abs=0.5)


@pytest.mark.skipif(
    not REAL_SURVEYOR_DXF.exists(),
    reason="real INGUR surveyor DXF not present",
)
def test_extract_real_surveyor():
    """The real INGUR surveyor DXF parses into the expected structure."""
    surv = extract_surveyor(REAL_SURVEYOR_DXF)
    surv.build_index()

    # ~522 boundary stones (B/BS/RBS/VBS/RB/RS), per the field file.
    assert 500 < len(surv.stones) < 540
    assert all(s.code in BOUNDARY_CODES for s in surv.stones)

    # 130 SITE DATA LINE polylines traced into chains.
    assert len(surv.polylines) == 130
    assert len(surv.chains) > 400

    # Coordinates sit in UTM Zone 44N (Tamil Nadu).
    xmin, ymin, xmax, ymax = surv.extent
    assert 600000 < xmin < 900000
    assert 1100000 < ymin < 1400000
    # ~2 km x ~3 km tower corridor.
    assert 1500 < (xmax - xmin) < 2500
    assert 2500 < (ymax - ymin) < 3500
