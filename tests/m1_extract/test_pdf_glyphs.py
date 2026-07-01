"""Tests for per-element glyph extraction from FMB PDF vector paths.

These tests cover the path-extraction and canvas-rendering steps without
invoking the OCR engine — making them fast (no model load). The correctness
oracle is structural: glyph groups must be found, canvases must be the right
shape, polygon orientation must round-trip through glyph_polygon correctly.
"""

from __future__ import annotations

import math
from pathlib import Path

import fitz
import numpy as np
import pytest

from landintel.pipeline.m1_extract.pdf_glyphs import (
    CANVAS_H,
    CANVAS_W,
    GlyphGroup,
    extract_glyph_groups,
    glyph_polygon,
    render_glyph_group,
)

FMB_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "FMB"


@pytest.fixture(scope="module")
def survey100_groups() -> list[GlyphGroup]:
    pdf = FMB_DIR / "FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_100.pdf"
    with fitz.open(str(pdf)) as doc:
        return extract_glyph_groups(doc[0])


def test_glyph_groups_found_in_real_fixture(survey100_groups: list[GlyphGroup]) -> None:
    """Glyph extraction captures all colored text fills — not just near-black.

    Survey 100 encodes ALL dimension numbers as colored vector filled paths:
      ~56 blue fills = dimension measurement labels
      ~17 red fills  = stone labels (A/B/C) and neighbor survey numbers
      ~5  black fills = some boundary measurement labels
    Total: ~78 text-sized fills. The previous near-black filter captured only 5.
    This test guards against regression back to a color-restricted filter.
    """
    # ~78 individual fills cluster into ~34 groups (multi-digit numbers have
    # multiple fills that merge). Old near-black filter gave only 5.
    assert len(survey100_groups) >= 25, (
        f"Expected >=25 glyph groups (all colors) in survey 100, got {len(survey100_groups)}. "
        "If this is <10 the color filter has regressed to near-black-only."
    )
    # Verify multi-color capture: should find both blue and red groups.
    colors = {g.color for g in survey100_groups}
    assert "blue" in colors, "Expected blue (dimension) fills — color filter may be broken"
    assert "red" in colors, "Expected red (stone label) fills — color filter may be broken"


def test_glyph_groups_have_valid_positions(survey100_groups: list[GlyphGroup]) -> None:
    """All group centres are inside the A4 page bounds (595 x 841 pt)."""
    for g in survey100_groups:
        cx, cy = g.center
        assert 0 <= cx <= 595, f"x out of page: {cx}"
        assert 0 <= cy <= 841, f"y out of page: {cy}"


def test_glyph_group_angle_in_range(survey100_groups: list[GlyphGroup]) -> None:
    """Angles are in [0, 180) — matching anchor._orientation convention."""
    for g in survey100_groups:
        assert 0.0 <= g.angle_deg < 180.0, f"angle out of range: {g.angle_deg}"


def test_render_produces_correct_canvas_shape(survey100_groups: list[GlyphGroup]) -> None:
    """Rendering any group produces a grayscale uint8 canvas of the expected size."""
    # Render the first few groups to stay fast.
    for g in survey100_groups[:5]:
        img = render_glyph_group(g)
        assert img.shape == (CANVAS_H, CANVAS_W), f"bad shape: {img.shape}"
        assert img.dtype == np.uint8
        # Canvas should not be entirely black (fill renders onto white).
        assert img.max() == 255, "canvas background should be white"


def test_render_produces_ink_on_canvas(survey100_groups: list[GlyphGroup]) -> None:
    """At least some groups produce dark pixels (the glyph fill was rendered)."""
    rendered_with_ink = 0
    for g in survey100_groups[:10]:
        img = render_glyph_group(g)
        if img.min() < 200:  # dark pixels present
            rendered_with_ink += 1
    # If the PDF does encode digits as vector paths, most renders have ink.
    # If all canvases are blank white, the PDF format is unexpected — still not
    # a crash, but we want to know.
    assert rendered_with_ink >= 1, (
        "Expected at least 1 group to produce ink; check if this PDF encodes "
        "dimension numbers as raster images rather than filled vector paths."
    )


def test_glyph_polygon_encodes_angle_correctly() -> None:
    """polygon[0]->polygon[1] direction matches the group's angle_deg."""
    for angle in (0.0, 30.0, 45.0, 70.0, 89.0):
        bbox = fitz.Rect(100, 200, 130, 210)  # 30x10 rectangle
        group = GlyphGroup(
            drawings=[],
            bbox=bbox,
            center=(115.0, 205.0),
            angle_deg=angle,
        )
        poly = glyph_polygon(group)
        assert len(poly) == 4

        # Direction of the top edge (p0->p1) should equal angle_deg.
        dx = poly[1][0] - poly[0][0]
        dy = poly[1][1] - poly[0][1]
        computed = math.degrees(math.atan2(dy, dx)) % 180.0
        assert abs(computed - angle) < 0.5, (
            f"angle_deg={angle} but polygon top-edge direction={computed:.2f}"
        )


def test_render_empty_drawings_group() -> None:
    """A group with no drawable items returns a valid all-white canvas."""
    bbox = fitz.Rect(100, 100, 130, 120)
    group = GlyphGroup(drawings=[{"items": []}], bbox=bbox, center=(115.0, 110.0), angle_deg=0.0)
    img = render_glyph_group(group)
    assert img.shape == (CANVAS_H, CANVAS_W)
    assert img.max() == 255
