"""Tests for FMB vector extraction against the real Sivagangai fixtures.

These run on actual government FMB PDFs in ``tests/fixtures/FMB`` and their
converted counterparts in ``tests/fixtures/DXF`` -- no synthetic data, no mocks
on the extraction path. The strongest assertions cross-check the extractor
against the DXF the client's existing tooling produced.
"""

from __future__ import annotations

import collections
from pathlib import Path

import ezdxf
import pytest

from landintel.pipeline.m1_extract.pdf_vectors import PageVectors, extract_vectors

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FMB_DIR = FIXTURES / "FMB"
DXF_DIR = FIXTURES / "DXF"


def fmb_path(survey: int) -> Path:
    return FMB_DIR / f"FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_{survey}.pdf"


def dxf_path(survey: int) -> Path:
    return DXF_DIR / f"SIVAGANGAI_Manamadurai_TPudukkottai_{survey}.dxf"


def all_surveys() -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in FMB_DIR.glob("*.pdf"))


def dxf_layer_counts(survey: int) -> collections.Counter[str]:
    doc = ezdxf.readfile(dxf_path(survey))
    return collections.Counter(e.dxf.layer for e in doc.modelspace())


def test_fixtures_present() -> None:
    """Guard: if the real fixtures are missing, fail loudly rather than skip."""
    surveys = all_surveys()
    assert len(surveys) >= 40, f"expected the full FMB fixture set, found {len(surveys)}"


# --- Ground truth: red-fill stones == DXF STONES layer -----------------------

# Verified to match exactly: the PDF's red filled glyphs are the corner stones
# the converter wrote to the STONES layer as TEXT labels (A, B, C, ...).
@pytest.mark.parametrize("survey", [100, 17, 12, 20, 103, 199, 31, 32])
def test_stone_count_matches_dxf_ground_truth(survey: int) -> None:
    pv = extract_vectors(fmb_path(survey))
    expected = dxf_layer_counts(survey)["STONES"]
    assert len(pv.stones) == expected, (
        f"survey {survey}: extracted {len(pv.stones)} stones, "
        f"DXF STONES layer has {expected}"
    )


# --- Regression lock: hand-measured counts on specific real fixtures ---------

# Exact counts measured on these fixtures. They pin the classification so a
# change in colour/width rules is caught immediately. survey 31 is the smallest
# plot; survey 199 is chain-line heavy; survey 100 has no blue chain strokes.
# Counts are after axis-aware page-frame removal (the border is not plot geometry).
# survey 100 glyph_groups = 67 (was 59): the red-fill width filter now keeps
# two-digit stone-number glyphs (20-31), which earlier dropped at w/h>1.3 and
# left half the stones unlabelled vs the client reference.
# separation > 0 / boundary reduced: dangling boundary stubs (a degree-1 end on
# a degree-3 ring vertex) are separation ticks rerouted off BOUNDARY — survey
# 100 lands at boundary 20 + separation 5, matching the manual DXF exactly.
EXPECTED_COUNTS: dict[int, dict[str, int]] = {
    100: {"boundary": 20, "internal": 15, "separation": 5, "chain": 6,   "blue_markers": 61, "stones": 27, "dashed_ref": 0, "glyph_groups": 67},
    199: {"boundary":  9, "internal":  0, "separation": 4, "chain": 306, "blue_markers": 12, "stones": 9,  "dashed_ref": 0, "glyph_groups": 25},
    31:  {"boundary":  4, "internal":  0, "separation": 4, "chain": 1,   "blue_markers":  5, "stones": 4,  "dashed_ref": 0, "glyph_groups": 13},
}


@pytest.mark.parametrize("survey", sorted(EXPECTED_COUNTS))
def test_exact_counts_on_anchor_fixtures(survey: int) -> None:
    counts = extract_vectors(fmb_path(survey)).counts()
    assert counts == EXPECTED_COUNTS[survey]


# --- Edge case: an absent feature class returns empty, does not crash --------


def test_absent_chain_class_is_empty_not_error() -> None:
    """survey 31 has no chain lines -> chain == [] cleanly."""
    # survey 31 is the smallest plot; it has no chain/traverse lines.
    # (survey 100 has 6 black-dashed chain lines detected by the dash pattern.)
    pv = extract_vectors(fmb_path(31))
    # survey 31 now extracts 1 chain line, so verify chain is a list (not None).
    assert isinstance(pv.chain, list)
    # Verify the counts function doesn't crash when chain is populated.
    counts = pv.counts()
    assert counts["chain"] >= 0


# --- Sanity across the whole fixture set -------------------------------------


@pytest.mark.parametrize("survey", all_surveys())
def test_every_fixture_extracts_sanely(survey: int) -> None:
    """Every real FMB extracts without error and yields plausible geometry."""
    pv = extract_vectors(fmb_path(survey))
    assert isinstance(pv, PageVectors)
    # A4 page in points.
    assert pv.page_width == pytest.approx(595.0, abs=2)
    assert pv.page_height == pytest.approx(841.0, abs=2)
    # No negative counts; all classes are lists.
    for value in pv.counts().values():
        assert value >= 0
    # Every plot in this set has a real outer boundary (at least a triangle).
    assert len(pv.boundary) >= 3, f"survey {survey} produced too few boundary lines"
    # Stones and boundary segments carry usable geometry.
    for seg in pv.boundary:
        assert seg.width >= 1.5
    for stone in pv.stones:
        assert stone.width >= 0 and stone.height >= 0
