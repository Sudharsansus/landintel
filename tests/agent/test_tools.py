"""Tests for the agent tools -- pure, deterministic, no API."""

from __future__ import annotations

import pytest

from landintel.agent.tools import (
    DEFAULT_TOOLS,
    check_area,
    dispatch,
    flag_for_review,
    validate_ocr,
)


@pytest.mark.parametrize(
    ("raw", "normalized", "valid", "reason"),
    [
        ("44,2", 44.2, True, "ok"),         # comma decimal
        ("(60,6)", 60.6, True, "ok"),        # chain-line parentheses
        ("5,", 5.0, True, "ok"),             # trailing separator
        ("145,6", 145.6, True, "ok"),
        ("Y8t6", None, False, "noise"),      # stray letters
        ("|0", None, False, "noise"),        # symbol
        ("13-11-2017", None, False, "noise"),  # a date, not a measurement
        ("2000", 2000.0, False, "out_of_range"),  # implausibly large
    ],
)
def test_validate_ocr(raw: str, normalized: float | None, valid: bool, reason: str) -> None:
    out = validate_ocr(raw=raw)
    assert out["normalized"] == normalized
    assert out["valid"] is valid
    assert out["reason"] == reason
    assert out["raw"] == raw  # raw preserved verbatim


def test_check_area_within_tolerance() -> None:
    # Survey 100: 16972 m^2 computed vs 1.665 ha stated -> ~1.9% error.
    out = check_area(computed_area_m2=16972.0, stated_area_ha=1.665)
    assert out["within_tolerance"] is True
    assert out["error"] == pytest.approx(0.0193, abs=0.002)


def test_check_area_out_of_tolerance() -> None:
    out = check_area(computed_area_m2=20000.0, stated_area_ha=1.0)
    assert out["within_tolerance"] is False
    assert out["error"] == pytest.approx(1.0)


def test_check_area_no_stated() -> None:
    out = check_area(computed_area_m2=100.0, stated_area_ha=0.0)
    assert out["within_tolerance"] is False
    assert out["reason"] == "no_stated_area"


def test_flag_for_review() -> None:
    out = flag_for_review(reason="area off by 7%", field="area", severity="high")
    assert out == {"flagged": True, "reason": "area off by 7%", "field": "area", "severity": "high"}


def test_dispatch_routes_to_handler() -> None:
    out = dispatch(DEFAULT_TOOLS, "validate_ocr", {"raw": "44,2"})
    assert out["normalized"] == 44.2


def test_dispatch_unknown_tool_raises() -> None:
    with pytest.raises(KeyError):
        dispatch(DEFAULT_TOOLS, "does_not_exist", {})


def test_tool_specs_shape() -> None:
    for tool in DEFAULT_TOOLS:
        spec = tool.spec()
        assert set(spec) == {"name", "description", "input_schema"}
        assert spec["input_schema"]["type"] == "object"
