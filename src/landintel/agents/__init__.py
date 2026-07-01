"""LandIntel runtime agent layer.

Four agents run automatically on every job so the shipped product is self-verifying
without a human (or Claude session) in the loop:

  * VerificationAgent -- asserts FP-safety + per-module invariants; FAIL -> don't ship
  * GuardAgent        -- deterministic final-line FP guards (e.g. 180-deg anagram trap)
  * InputRequestAgent -- the MINIMAL extra input that closes each unplaced plot -> 100%
  * LLMAssistAgent    -- local open-source LLM + Claude combo for fuzzy tasks; never decides

THE INVARIANT: agents only verify / request input / narrate. ACCEPT is decided solely by
the deterministic math gates, so no agent can ever produce a false positive.
"""

from .base import Agent, AgentReport, Check, InputRequest, InputType, Severity
from .dispositions import (
    CONFIDENT, VALID_STATES, PlotDisposition, from_club_result, from_georef_result,
    normalize,
)
from .orchestrator import run_agent_layer

__all__ = [
    "Agent", "AgentReport", "Check", "InputRequest", "InputType", "Severity",
    "run_agent_layer",
    "PlotDisposition", "from_club_result", "from_georef_result", "normalize",
    "CONFIDENT", "VALID_STATES",
]
