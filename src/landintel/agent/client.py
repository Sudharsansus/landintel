"""Thin Anthropic wrapper: model from config, retries, and a tool-use loop.

Plumbing, not a framework. It runs one agent task -- system prompt + user message
-- through the Messages API, dispatching any tool calls to the deterministic
handlers in :mod:`tools` and feeding the results back until the model stops. It
returns a structured :class:`AgentResult` the pipeline can act on, and raises
:class:`AgentError` on hard failures (transient errors exhausted, or the loop not
converging). No geometry, no OCR, no judgement -- those live in the agent modules
that call this.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from ..config import get_settings
from ..core.exceptions import AgentError
from .tools import Tool, dispatch, tool_specs

__all__ = ["AgentClient", "AgentResult", "ToolCall", "load_prompt"]

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Transient API failures worth retrying with backoff.
_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def load_prompt(name: str) -> str:
    """Load a versioned system prompt from ``agent/prompts/<name>.txt``."""
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise AgentError("prompt not found", prompt=name, path=str(path))
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class ToolCall:
    """One tool the agent invoked, with the input it passed and the result."""

    name: str
    input: dict[str, Any]
    result: dict[str, Any]
    is_error: bool = False


@dataclass
class AgentResult:
    """The outcome of an agent run."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    turns: int = 0


class AgentClient:
    """Runs agent tasks against the Anthropic Messages API.

    Construct with the configured model/key by default, or inject a client and
    model for testing. ``tools`` are the tools this client exposes to the model.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        api_key: str | None = None,
        tools: list[Tool] | None = None,
        max_tokens: int = 1024,
        max_retries: int = 4,
        retry_base_delay: float = 0.5,
        max_turns: int = 8,
    ) -> None:
        if client is None:
            settings = get_settings()
            client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)
            model = model or settings.anthropic_model
        if model is None:
            model = get_settings().anthropic_model
        self._client = client
        self.model = model
        self.tools = tools or []
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.max_turns = max_turns

    def _create(self, system: str, messages: list[dict[str, Any]]) -> Any:
        """One Messages API call, retried with exponential backoff on transients."""
        specs = tool_specs(self.tools)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=messages,
                    tools=specs,
                )
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_base_delay * (2**attempt))
        raise AgentError(
            "Anthropic call failed after retries",
            attempts=self.max_retries,
        ) from last_exc

    def run(self, *, system: str, user: str) -> AgentResult:
        """Run the tool-use loop until the model stops, and return the result."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        tool_calls: list[ToolCall] = []

        for turn in range(1, self.max_turns + 1):
            response = self._create(system, messages)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return AgentResult(
                    text=_text_of(response.content),
                    tool_calls=tool_calls,
                    stop_reason=response.stop_reason,
                    turns=turn,
                )

            results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                call = self._run_tool(block)
                tool_calls.append(call)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(call.result),
                        "is_error": call.is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

        raise AgentError("agent did not converge", max_turns=self.max_turns)

    def _run_tool(self, block: Any) -> ToolCall:
        """Dispatch one tool_use block, capturing handler errors as tool results."""
        try:
            result = dispatch(self.tools, block.name, dict(block.input))
            return ToolCall(name=block.name, input=dict(block.input), result=result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a result
            return ToolCall(
                name=block.name,
                input=dict(block.input),
                result={"error": str(exc)},
                is_error=True,
            )


def _text_of(content: list[Any]) -> str:
    """Concatenate the text blocks of a response's content."""
    parts = [
        block.text
        for block in content
        if getattr(block, "type", None) == "text"
    ]
    return "\n".join(parts).strip()
