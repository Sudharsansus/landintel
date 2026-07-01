"""Resolve a Plot's raw OCR measurements into normalized values or flags.

Single responsibility: fill the ``value=None`` gap ``build_plot`` deliberately
left. Each measurement becomes either a normalized number (accepted) or a flag
on the plot (uncertain / unreadable). Nothing else.

The cheap part is deterministic and stays out of Claude: turning "44,2" into
44.2 is a rule, not a judgement, so a clean, in-range, in-confidence reading is
normalized in plain Python with no API call. Claude is asked only about the
genuinely ambiguous ones -- garbled tokens, implausible values, or low-confidence
detections -- and they are sent in a single batched call, not one per reading.

Routing (the OCRFailure-vs-ValidationFlag boundary from ``core.exceptions``, made
concrete):
- clean number, in plausible range, fits the plot's size envelope, confidence
  above threshold -> ACCEPT deterministically.
- anything else -> escalate to Claude, which decides accept / flag / reject. With
  no client available it is flagged (degraded mode never silently invents a value).

The plot is mutated in place: accepted measurements get their ``value`` set;
uncertain ones add a reason to ``plot.flags``; ``plot.status`` ends ``FLAGGED`` if
anything was flagged, else ``VALIDATED``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..core.enums import PlotStatus
from ..core.models import Measurement, Plot
from .client import AgentClient, load_prompt
from .tools import validate_ocr

__all__ = ["validate_plot", "ValidationOutcome", "HIGH_CONFIDENCE"]

HIGH_CONFIDENCE = 0.85
"""At/above this OCR confidence a clean, plausible reading is auto-accepted."""

ENVELOPE_FACTOR = 1.5
"""A reading larger than this multiple of the plot's span is implausible."""

MIN_MEASUREMENT_M = 0.1
MAX_MEASUREMENT_M = 1000.0


@dataclass
class ValidationOutcome:
    """Summary of a validation pass over one plot."""

    accepted: int = 0
    flagged: int = 0
    rejected: int = 0
    escalated: int = 0
    """How many ambiguous readings were sent to Claude."""


def _plot_span(plot: Plot) -> float | None:
    """The plot's largest dimension in metres, or ``None`` if no boundary."""
    if plot.boundary is None or not plot.boundary.points:
        return None
    xs = [p[0] for p in plot.boundary.points]
    ys = [p[1] for p in plot.boundary.points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _deterministic(measurement: Measurement, span: float | None) -> tuple[bool, float | str]:
    """Decide a reading without Claude.

    Returns ``(True, value)`` to accept, or ``(False, reason)`` to escalate.
    """
    result = validate_ocr(
        raw=measurement.raw, min_value=MIN_MEASUREMENT_M, max_value=MAX_MEASUREMENT_M
    )
    normalized = result["normalized"]
    if normalized is None:
        return False, "noise"
    if not result["valid"]:
        return False, "out_of_range"
    if span is not None and normalized > ENVELOPE_FACTOR * span:
        return False, "exceeds_plot_span"
    if measurement.confidence < HIGH_CONFIDENCE:
        return False, "low_confidence"
    return True, normalized


def _flag_text(measurement: Measurement, reason: str) -> str:
    """Human-readable flag entry tying the reason to the specific reading."""
    return f"measurement {measurement.raw!r} (conf {measurement.confidence:.2f}): {reason}"


def _build_escalation_prompt(
    plot: Plot, ambiguous: list[tuple[int, Measurement, str]]
) -> str:
    """One batched user message describing the plot and the uncertain readings."""
    span = _plot_span(plot)
    readings = [
        {
            "id": index,
            "raw": measurement.raw,
            "confidence": round(measurement.confidence, 2),
            "hint": reason,
        }
        for index, measurement, reason in ambiguous
    ]
    context = {
        "survey_no": plot.survey_no,
        "stated_area_ha": plot.stated_area,
        "plot_span_m": round(span, 1) if span is not None else None,
    }
    return (
        "Resolve these OCR readings for the survey below. Context:\n"
        + json.dumps(context)
        + "\nReadings:\n"
        + json.dumps(readings)
    )


def _parse_decisions(text: str) -> dict[int, dict]:
    """Parse Claude's JSON decision array; tolerate code fences / stray prose.

    Returns ``{id: decision}``. An unparseable response yields ``{}`` so the
    caller falls back to flagging -- never to inventing values.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return {}
    try:
        decisions = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}
    return {d["id"]: d for d in decisions if isinstance(d, dict) and "id" in d}


def _escalate(
    plot: Plot, ambiguous: list[tuple[int, Measurement, str]], client: AgentClient
) -> dict[int, dict]:
    """Ask Claude to resolve the ambiguous readings; return decisions by id."""
    result = client.run(
        system=load_prompt("validate"),
        user=_build_escalation_prompt(plot, ambiguous),
    )
    return _parse_decisions(result.text)


def validate_plot(plot: Plot, *, client: AgentClient | None = None) -> ValidationOutcome:
    """Normalize the plot's measurements, flagging the uncertain ones.

    Args:
        plot: The plot to validate, mutated in place.
        client: An agent client for escalating ambiguous readings. When ``None``,
            ambiguous readings are flagged rather than escalated (degraded mode).

    Returns:
        A :class:`ValidationOutcome` counting accepted / flagged / rejected /
        escalated readings.
    """
    span = _plot_span(plot)
    outcome = ValidationOutcome()
    ambiguous: list[tuple[int, Measurement, str]] = []

    for index, measurement in enumerate(plot.measurements):
        accept, payload = _deterministic(measurement, span)
        if accept:
            measurement.value = float(payload)
            outcome.accepted += 1
        else:
            ambiguous.append((index, measurement, str(payload)))

    if ambiguous and client is not None:
        outcome.escalated = len(ambiguous)
        decisions = _escalate(plot, ambiguous, client)
        for index, measurement, hint in ambiguous:
            decision = decisions.get(index, {})
            verdict = decision.get("decision", "flag")
            value = decision.get("value")
            if verdict == "accept" and value is not None:
                measurement.value = float(value)
                outcome.accepted += 1
            elif verdict == "reject":
                outcome.rejected += 1  # leave value None; it was not a measurement
            else:  # "flag" or anything unrecognized -> human review
                plot.flags.append(_flag_text(measurement, decision.get("reason", hint)))
                outcome.flagged += 1
    else:
        for _index, measurement, reason in ambiguous:
            plot.flags.append(_flag_text(measurement, reason))
            outcome.flagged += 1

    plot.status = PlotStatus.FLAGGED if plot.flags else PlotStatus.VALIDATED
    return outcome
