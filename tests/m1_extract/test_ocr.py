"""Tests for PP-OCRv5 extraction against a real FMB fixture.

Runs the actual OCR engine (no mocks on the OCR path) on survey 100 and checks
its output against ground truth taken from the matching DXF's dimension text.

Reality being measured, not wished for: FMB dimension numbers are small and
rotated along survey lines, so the local *mobile* models read only a fraction
of them on a single pass (the near-horizontal ones cleanly, the steeply-rotated
ones poorly). These tests therefore assert a *curated, verified* set of
measurements is found -- proving real reading of real values -- rather than
claiming full recall. Production uses the server detection model for better
recall; the threshold here reflects the mobile default.

The OCR engine init + first inference is slow, so results are computed once per
fixture via module-scoped fixtures and shared across assertions.
"""

from __future__ import annotations

import re
from pathlib import Path

import ezdxf
import pytest

from landintel.pipeline.m1_extract.ocr import (
    OCRDetection,
    extract_text,
    parse_header,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FMB_DIR = FIXTURES / "FMB"
DXF_DIR = FIXTURES / "DXF"

# Page size of the FMB sheets (A4 in points); positions must fall inside this.
PAGE_W, PAGE_H = 595.0, 841.0

MIN_CONF = 0.85

# Measurements present in survey 100's DXF dimension layers AND reliably read by
# the mobile model at high confidence. Used as positive anchors: enough must be
# found to prove OCR is genuinely reading the survey's real numbers.
# NOTE: mobile OCR recall is ~24% on rotated dimension text. The values below
# are from BOUNDARY_DIMENSIONS (near-horizontal labels read cleanly). DIMENSIONS
# layer values (16.4, 25.8, 28.6 etc. on rotated subdivision lines) are NOT
# reliably read by mobile OCR and are excluded from this anchor set.
ANCHOR_MEASUREMENTS = [
    "34.6", "46.6", "51.4", "69.0", "78.0",
]
MIN_ANCHORS_FOUND = 3


def fmb_path(survey: int) -> Path:
    return FMB_DIR / f"FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_{survey}.pdf"


def canon(text: str) -> str:
    """Canonicalize a measurement for comparison: drop parens/space, comma->dot.

    Test-side only -- the source uses comma decimals and OCR may emit either
    separator; ``ocr.py`` itself returns the raw string untouched.
    """
    return re.sub(r"[()\s]", "", text).replace(",", ".")


@pytest.fixture(scope="module")
def survey100() -> list[OCRDetection]:
    return extract_text(fmb_path(100))


@pytest.fixture(scope="module")
def survey100_known() -> set[str]:
    """Canonical dimension values from survey 100's DXF (ground truth)."""
    doc = ezdxf.readfile(DXF_DIR / "SIVAGANGAI_Manamadurai_TPudukkottai_100.dxf")
    layers = {"BOUNDARY_DIMENSIONS", "DIMENSIONS", "CHAINLINE_DIMENSIONS"}
    return {
        canon(e.dxf.text)
        for e in doc.modelspace()
        if e.dxf.layer in layers and e.dxftype() == "TEXT"
    }


def test_ocr_returns_many_detections(survey100: list[OCRDetection]) -> None:
    # After the header/body split: header_dets (~10-20) + body_glyph_dets (~48) = ~58+.
    # Lower threshold reflects real clean signal, not page-OCR garbage flood.
    assert len(survey100) >= 40
    numeric = [d for d in survey100 if re.search(r"\d", d.text)]
    assert len(numeric) >= 30


def test_header_metadata_read_correctly(survey100: list[OCRDetection]) -> None:
    """The header block (district/taluk/village) matches the fixture's identity."""
    blob = " | ".join(d.text for d in survey100 if d.confidence >= MIN_CONF)
    assert "Sivagangai" in blob
    assert "Manamadurai" in blob
    assert "Pudukkottai" in blob


def test_known_measurements_found_above_confidence(
    survey100: list[OCRDetection],
) -> None:
    """A curated set of real measurements is read at high confidence."""
    read = {canon(d.text) for d in survey100 if d.confidence >= MIN_CONF}
    found = [m for m in ANCHOR_MEASUREMENTS if m in read]
    assert len(found) >= MIN_ANCHORS_FOUND, (
        f"only found {found} of anchor set {ANCHOR_MEASUREMENTS}"
    )


def test_found_measurements_are_ground_truth(
    survey100: list[OCRDetection], survey100_known: set[str]
) -> None:
    """Every anchor we assert on genuinely exists in the DXF ground truth."""
    assert set(ANCHOR_MEASUREMENTS) <= survey100_known


def test_detection_contract(survey100: list[OCRDetection]) -> None:
    """Each detection carries valid raw text, confidence, and in-bounds position."""
    for d in survey100:
        assert isinstance(d.text, str) and d.text != ""
        assert 0.0 <= d.confidence <= 1.0
        assert len(d.polygon) >= 3  # detection boxes are quads
        cx, cy = d.center
        # positions are in PDF page space (pixels / zoom), so within the page
        assert -1.0 <= cx <= PAGE_W + 1.0
        assert -1.0 <= cy <= PAGE_H + 1.0


def test_header_parses_scale_and_area(survey100: list[OCRDetection]) -> None:
    """The scale and stated area -- the geometry-gating header fields -- parse."""
    header = parse_header(survey100)
    assert header.survey_no == "100"
    assert header.district == "Sivagangai"
    assert header.taluk == "Manamadurai"
    assert header.village.startswith("T.Pudukkottai")
    # "Scale : 1 : 2021" -> denominator 2021 (drives pixel->metre conversion).
    assert header.scale_denominator == 2021
    # "Area : Hect 01 Ares 66.50" -> 1 ha + 66.50 ares = 1.665 ha.
    assert header.stated_area_ha == pytest.approx(1.665)


def test_second_fixture_also_extracts(survey100: list[OCRDetection]) -> None:
    """Smoke a second, smaller plot to confirm extraction generalizes."""
    dets = extract_text(fmb_path(31))
    assert len(dets) >= 20
    blob = " | ".join(d.text for d in dets if d.confidence >= MIN_CONF)
    assert "Pudukkottai" in blob
