"""Tools the agent can call: deterministic operations, not judgement.

Each tool is a JSON schema (so Claude knows how to call it) plus a plain Python
handler the tool-use loop dispatches to. The handlers *compute* -- normalize and
range-check a reading, compare two areas, record a flag -- and return structured
results. Whether to accept, reject, or flag is Claude's decision, made from those
results; the principle is "the agent decides, the pipeline computes".

There is no geometry or OCR logic in here. The richer agent modules
(``validator.py``, ``anomaly.py``) orchestrate prompt + client + these tools and
apply the outcome to a Plot; this file is just the atomic verbs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Tool",
    "validate_ocr",
    "check_area",
    "flag_for_review",
    "DEFAULT_TOOLS",
    "tool_specs",
    "dispatch",
]


@dataclass(frozen=True)
class Tool:
    """A callable tool: its Anthropic schema plus the handler to run."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def spec(self) -> dict[str, Any]:
        """The tool definition in the shape the Anthropic Messages API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# --- Handlers ----------------------------------------------------------------

# A clean measurement, once parentheses/spaces are stripped and the comma decimal
# is turned into a dot. Anything with stray letters/symbols fails this and is noise.
_CLEAN_NUMBER = re.compile(r"-?\d+(\.\d+)?")


def validate_ocr(
    raw: str, min_value: float = 0.1, max_value: float = 1000.0
) -> dict[str, Any]:
    """Normalize one OCR reading and range-check it. No judgement, just facts.

    Returns the normalized value when the token is a clean number, plus a
    ``reason`` of ``ok`` / ``out_of_range`` / ``noise`` the agent reasons about.
    """
    cleaned = raw.strip().strip("()").replace(",", ".").replace(" ", "").rstrip(".")
    if not _CLEAN_NUMBER.fullmatch(cleaned):
        return {"raw": raw, "normalized": None, "valid": False, "reason": "noise"}
    value = float(cleaned)
    in_range = min_value <= value <= max_value
    return {
        "raw": raw,
        "normalized": value,
        "valid": in_range,
        "reason": "ok" if in_range else "out_of_range",
    }


def check_area(
    computed_area_m2: float, stated_area_ha: float, tolerance: float = 0.05
) -> dict[str, Any]:
    """Compare a computed polygon area against the FMB stated area.

    ``computed_area_m2`` is square metres (from the scaled boundary);
    ``stated_area_ha`` is hectares (from the header). Returns the relative error
    and whether it is within ``tolerance``; the agent decides flag-vs-pass.
    """
    if stated_area_ha <= 0:
        return {"within_tolerance": False, "reason": "no_stated_area"}
    computed_ha = computed_area_m2 / 10_000.0
    error = abs(computed_ha - stated_area_ha) / stated_area_ha
    return {
        "computed_ha": computed_ha,
        "stated_ha": stated_area_ha,
        "ratio": computed_ha / stated_area_ha,
        "error": error,
        "within_tolerance": error <= tolerance,
    }


def flag_for_review(
    reason: str, field: str | None = None, severity: str = "medium"
) -> dict[str, Any]:
    """Record a human-review flag. Returns the structured flag (does not raise)."""
    return {"flagged": True, "reason": reason, "field": field, "severity": severity}


# --- Tool definitions --------------------------------------------------------

_VALIDATE_OCR = Tool(
    name="validate_ocr",
    description=(
        "Normalize an OCR reading (comma decimal -> dot, strip parentheses) and "
        "check it falls in a plausible FMB measurement range. Returns the "
        "normalized value or marks it as noise/out-of-range."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "raw": {"type": "string", "description": "The raw OCR text, untouched."},
            "min_value": {"type": "number", "description": "Lower plausible bound (m)."},
            "max_value": {"type": "number", "description": "Upper plausible bound (m)."},
        },
        "required": ["raw"],
    },
    handler=validate_ocr,
)

_CHECK_AREA = Tool(
    name="check_area",
    description=(
        "Compare a computed polygon area (square metres) against the FMB "
        "stated area (hectares); returns the relative error and whether it is "
        "within tolerance."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "computed_area_m2": {"type": "number"},
            "stated_area_ha": {"type": "number"},
            "tolerance": {"type": "number", "description": "Relative tolerance, e.g. 0.05."},
        },
        "required": ["computed_area_m2", "stated_area_ha"],
    },
    handler=check_area,
)

_FLAG_FOR_REVIEW = Tool(
    name="flag_for_review",
    description=(
        "Flag a plot or reading for human review when it is plausible but "
        "uncertain. Does not fail the job; records the reason and severity."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "field": {"type": "string", "description": "The field in question, if specific."},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["reason"],
    },
    handler=flag_for_review,
)

DEFAULT_TOOLS: list[Tool] = [_VALIDATE_OCR, _CHECK_AREA, _FLAG_FOR_REVIEW]


def tool_specs(tools: list[Tool]) -> list[dict[str, Any]]:
    """The Anthropic tool specs for a set of tools."""
    return [tool.spec() for tool in tools]


def dispatch(tools: list[Tool], name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Run the named tool's handler with ``tool_input``.

    Raises ``KeyError`` if no such tool is registered -- the caller turns that
    into an error tool-result so the agent can recover.
    """
    by_name = {tool.name: tool for tool in tools}
    return by_name[name].handler(**tool_input)
