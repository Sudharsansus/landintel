"""Tests for the agent client's tool-use loop and retry logic.

The Anthropic API itself is external, so the client is exercised with a fake
that returns scripted responses (and scripted transient errors). This tests our
loop/retry/dispatch logic, not Claude.
"""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from landintel.agent.client import AgentClient, load_prompt
from landintel.agent.tools import DEFAULT_TOOLS
from landintel.core.exceptions import AgentError


# --- Fakes -------------------------------------------------------------------


class FakeMessages:
    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        # Snapshot the messages list: run() reuses and appends to the same list,
        # so storing it by reference would show a later state.
        snapshot = {**kwargs, "messages": list(kwargs.get("messages", []))}
        self.calls.append(snapshot)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script: list) -> None:
        self.messages = FakeMessages(script)


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def tool_block(block_id: str, name: str, inp: dict):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=inp)


def response(stop_reason: str, blocks: list):
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


def make_client(script: list, **kwargs) -> AgentClient:
    return AgentClient(
        client=FakeClient(script),
        model="test-model",
        tools=DEFAULT_TOOLS,
        retry_base_delay=0.0,
        **kwargs,
    )


# --- Tool-use loop -----------------------------------------------------------


def test_tool_use_loop_dispatches_and_finishes() -> None:
    script = [
        response("tool_use", [tool_block("t1", "validate_ocr", {"raw": "44,2"})]),
        response("end_turn", [text_block("Validated: 44.2 is OK.")]),
    ]
    client = make_client(script)
    result = client.run(system="sys", user="validate 44,2")

    assert result.stop_reason == "end_turn"
    assert result.turns == 2
    assert result.text == "Validated: 44.2 is OK."
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "validate_ocr"
    assert call.result["normalized"] == 44.2
    assert call.is_error is False


def test_tool_results_are_fed_back_to_model() -> None:
    """The second API call must include the tool_result the loop produced."""
    fake = FakeClient(
        [
            response("tool_use", [tool_block("t1", "validate_ocr", {"raw": "44,2"})]),
            response("end_turn", [text_block("done")]),
        ]
    )
    client = AgentClient(client=fake, model="test-model", tools=DEFAULT_TOOLS, retry_base_delay=0.0)
    client.run(system="sys", user="go")

    second_call_messages = fake.messages.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "t1"


def test_unknown_tool_returns_error_result_not_crash() -> None:
    script = [
        response("tool_use", [tool_block("t1", "no_such_tool", {})]),
        response("end_turn", [text_block("recovered")]),
    ]
    result = make_client(script).run(system="sys", user="go")
    assert result.tool_calls[0].is_error is True
    assert "error" in result.tool_calls[0].result


# --- Retry -------------------------------------------------------------------


def test_retries_transient_error_then_succeeds() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    script = [
        anthropic.APIConnectionError(message="boom", request=request),
        response("end_turn", [text_block("ok after retry")]),
    ]
    client = make_client(script)
    result = client.run(system="sys", user="go")
    assert result.text == "ok after retry"
    assert len(client._client.messages.calls) == 2  # failed once, retried once


def test_retries_exhausted_raises_agent_error() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    script = [anthropic.APIConnectionError(message="boom", request=request)] * 3
    client = make_client(script, max_retries=3)
    with pytest.raises(AgentError):
        client.run(system="sys", user="go")


# --- Non-convergence ---------------------------------------------------------


def test_non_convergence_raises_agent_error() -> None:
    script = [
        response("tool_use", [tool_block("t1", "validate_ocr", {"raw": "1,0"})]),
        response("tool_use", [tool_block("t2", "validate_ocr", {"raw": "2,0"})]),
    ]
    client = make_client(script, max_turns=2)
    with pytest.raises(AgentError):
        client.run(system="sys", user="loop forever")


# --- Prompt loading ----------------------------------------------------------


def test_load_prompt_reads_validate() -> None:
    text = load_prompt("validate")
    assert "OCR validation agent" in text
    assert "44,2" in text  # the comma-decimal rule is documented


def test_load_prompt_missing_raises() -> None:
    with pytest.raises(AgentError):
        load_prompt("does_not_exist")
