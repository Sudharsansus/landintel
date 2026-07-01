"""Regression tests for the cross-village cadastral guard.

A FMB from a DIFFERENT village (e.g. KANDAMPALAYAM survey 9) must never be
cadastral-ACCEPTed onto this village's cadastre, even if a same-numbered parcel
here happens to fit. The corridor schedule cannot tell it apart (legitimate
off-corridor INGUR plots are off-schedule too), so the guard uses the FMB's own
village token from its filename. Measured on INGUR: without the guard, survey 9
produced a clean rigid fit (area 0.80, scale 1.09) -- a false positive.
"""
from __future__ import annotations

from landintel.pipeline.m2_georef.pipeline import _is_cross_village


def test_same_village_not_flagged():
    assert not _is_cross_village(
        "test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_INGUR_763.dxf", "INGUR")


def test_different_village_flagged():
    assert _is_cross_village(
        "test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_KANDAMPALAYAM _9.dxf", "INGUR")


def test_village_with_space_normalised():
    # Filename has a stray space before the survey number ("KANDAMPALAYAM _9").
    assert _is_cross_village("FMB_X_Y_KANDAMPALAYAM _9.dxf", "INGUR")
    assert not _is_cross_village("FMB_X_Y_INGUR _763.dxf", "INGUR")


def test_unparseable_is_conservative():
    # When the village can't be parsed, do NOT reject (no spurious FNs).
    assert not _is_cross_village("m1_synth_784.dxf", "INGUR")
    assert not _is_cross_village("anything.dxf", None)
