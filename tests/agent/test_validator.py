"""Tests for the OCR validator: deterministic normalization + escalation.

Per the agreed approach: the deterministic path runs with NO API call (proven by
a client that raises if touched); the escalation path is exercised with a fake
client returning scripted decisions; one live test runs only when an API key is
present.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from landintel.agent.client import AgentClient
from landintel.agent.tools import DEFAULT_TOOLS
from landintel.agent.validator import validate_plot
from landintel.core.enums import PlotStatus
from landintel.core.models import Boundary, Measurement, Plot
from landintel.pipeline.m1_extract.anchor import anchor_measurements
from landintel.pipeline.m1_extract.build_plot import build_plot
from landintel.pipeline.m1_extract.ocr import extract_text, parse_header
from landintel.pipeline.m1_extract.pdf_vectors import extract_vectors

FMB_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "FMB"

# A 100 m square, so readings up to ~150 m are within the plot envelope.
SQUARE = Boundary(points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)])


def make_plot(measurements: list[Measurement]) -> Plot:
    return Plot(
        client_id="c",
        survey_no="42",
        district="D",
        taluk="T",
        village="V",
        scale=2000,
        stated_area=1.0,
        boundary=SQUARE,
        measurements=measurements,
    )


class RaisingClient:
    """A client that must never be called -- proves the deterministic path is local."""

    def run(self, **kwargs):  # noqa: ANN003, ANN201
        raise AssertionError("Claude was called for a clean reading")


# --- Deterministic path (no API) ---------------------------------------------


def test_clean_readings_normalize_without_calling_claude() -> None:
    plot = make_plot(
        [
            Measurement(raw="44,2", confidence=0.95),
            Measurement(raw="(60,6)", confidence=0.99),
            Measurement(raw="145,6", confidence=0.90),
        ]
    )
    outcome = validate_plot(plot, client=RaisingClient())  # would raise if escalated

    assert outcome.accepted == 3
    assert outcome.escalated == 0 and outcome.flagged == 0
    assert [m.value for m in plot.measurements] == [44.2, 60.6, 145.6]
    assert plot.status is PlotStatus.VALIDATED


def test_ambiguous_readings_flag_when_no_client() -> None:
    plot = make_plot(
        [
            Measurement(raw="44,2", confidence=0.95),     # clean -> accept
            Measurement(raw="60,6", confidence=0.60),     # low confidence -> flag
            Measurement(raw="Y8t6", confidence=0.90),     # noise -> flag
            Measurement(raw="500,0", confidence=0.95),    # exceeds 100 m span -> flag
        ]
    )
    outcome = validate_plot(plot, client=None)

    assert outcome.accepted == 1
    assert outcome.flagged == 3
    assert plot.measurements[0].value == 44.2
    # Nothing uncertain was force-normalized into a fake value.
    assert all(m.value is None for m in plot.measurements[1:])
    assert plot.status is PlotStatus.FLAGGED
    assert len(plot.flags) == 3


# --- Escalation path (mocked Claude) -----------------------------------------


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        block = SimpleNamespace(type="text", text=self._text)
        return SimpleNamespace(stop_reason="end_turn", content=[block])


class _FakeAnthropic:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


def fake_agent_text(text: str) -> AgentClient:
    return AgentClient(
        client=_FakeAnthropic(text),
        model="test-model",
        tools=DEFAULT_TOOLS,
        retry_base_delay=0.0,
    )


def fake_agent(decisions: list[dict]) -> AgentClient:
    return fake_agent_text(json.dumps(decisions))


def test_escalation_applies_claude_decisions() -> None:
    plot = make_plot(
        [
            Measurement(raw="44,2", confidence=0.95),   # id 0: clean -> accept locally
            Measurement(raw="60,6", confidence=0.60),   # id 1: low conf -> escalate
            Measurement(raw="Y8t6", confidence=0.90),   # id 2: noise -> escalate
        ]
    )
    client = fake_agent(
        [
            {"id": 1, "decision": "accept", "value": 60.6, "reason": "clear on review"},
            {"id": 2, "decision": "reject", "value": None, "reason": "not a measurement"},
        ]
    )
    outcome = validate_plot(plot, client=client)

    assert outcome.escalated == 2
    assert outcome.accepted == 2          # one local, one via Claude
    assert outcome.rejected == 1
    assert outcome.flagged == 0
    assert plot.measurements[1].value == 60.6   # Claude accepted
    assert plot.measurements[2].value is None   # rejected, not invented
    assert plot.status is PlotStatus.VALIDATED


def test_escalation_flags_when_claude_says_flag() -> None:
    plot = make_plot([Measurement(raw="abc", confidence=0.5)])
    client = fake_agent([{"id": 0, "decision": "flag", "value": None, "reason": "unreadable"}])
    outcome = validate_plot(plot, client=client)
    assert outcome.flagged == 1
    assert plot.measurements[0].value is None
    assert plot.status is PlotStatus.FLAGGED
    assert "unreadable" in plot.flags[0]


def test_unparseable_claude_response_falls_back_to_flag() -> None:
    """A garbled model response flags the readings; it never invents values."""
    plot = make_plot([Measurement(raw="Y8t6", confidence=0.9)])
    client = fake_agent_text("sorry, I could not produce JSON")
    outcome = validate_plot(plot, client=client)
    assert outcome.flagged == 1
    assert plot.measurements[0].value is None


# --- Real fixture, deterministic only ----------------------------------------


def test_real_plot_normalizes_clean_and_never_fakes_noise() -> None:
    f = FMB_DIR / "FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_100.pdf"
    vectors = extract_vectors(f)
    detections = extract_text(f)
    plot = build_plot(
        client_id="c",
        vectors=vectors,
        detections=detections,
        anchor_result=anchor_measurements(vectors, detections),
        header=parse_header(detections),
    )
    outcome = validate_plot(plot, client=None)  # deterministic, no API

    assert outcome.accepted > 0
    accepted = {m.value for m in plot.measurements if m.value is not None}
    # Known clean boundary dimensions read at high confidence are normalized.
    # (69.0 and 51.4 are the two highest-confidence boundary dims on survey 100.)
    assert 69.0 in accepted and 51.4 in accepted
    # No non-numeric token was ever turned into a value (the core "no false positive").
    for m in plot.measurements:
        cleaned = m.raw.strip().strip("()").replace(",", ".").replace(" ", "").rstrip(".")
        if not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
            assert m.value is None, f"noise {m.raw!r} was force-normalized"


# --- Live integration (opt-in) -----------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="set ANTHROPIC_API_KEY to run the live agent test",
)
def test_live_escalation_round_trip() -> None:
    plot = make_plot(
        [
            Measurement(raw="44,2", confidence=0.95),
            Measurement(raw="Y8t6", confidence=0.55),
        ]
    )
    client = AgentClient(tools=DEFAULT_TOOLS)
    outcome = validate_plot(plot, client=client)
    assert outcome.escalated >= 1
    assert plot.measurements[0].value == 44.2  # clean one resolved locally
