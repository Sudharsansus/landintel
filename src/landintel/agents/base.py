"""Shared models for the LandIntel runtime agent layer.

The agent layer runs automatically on EVERY job so the shipped product catches its
own errors without a human (or Claude session) in the loop. THE HARD RULE: an agent
may only (a) flag a plot for review, (b) request a specific extra input, or (c)
annotate/explain -- it may NEVER promote a placement to ACCEPT. ACCEPT is decided
solely by the deterministic math gates, so no agent (and no LLM hallucination) can
ever create a false positive. The agents make the pipeline self-verifying and turn
"100% accuracy" into a process: place what is provably correct, and for the rest emit
the minimal extra input that closes the gap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    OK = "ok"
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"          # a job-blocking invariant violation -> must not ship


class InputType(str, Enum):
    """The minimal extra input that would close a plot to a confident placement."""
    TWO_CORNER_SEED = "two_corner_seed"      # operator gives 2 corner->UTM points (always FP-safe)
    CLEARER_PARCEL = "clearer_parcel"        # a clearer/closed parcel polygon (shapefile/KML/image)
    VILLAGE_REFERENCE = "village_reference"  # a different village's cadastral/surveyor reference
    CONFIRM_PLACEMENT = "confirm_placement"  # located but unconfirmed -> a human yes/no
    NONE = "none"                            # already confident; no input needed


@dataclass
class Check:
    """One invariant/assertion result from an agent."""
    name: str
    severity: Severity
    detail: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class InputRequest:
    """A precise, minimal request for the one extra input that closes a plot."""
    survey_number: str
    disposition: str
    reason: str                 # why it is not confident (the failing gate / data gap)
    input_type: InputType
    instruction: str            # human-readable ask
    resolves_via: str           # the pipeline path that consumes the input (e.g. seed_place)
    known_utm: tuple[float, float] | None = None  # best-known position, if any (a hint)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["input_type"] = self.input_type.value
        d["known_utm"] = list(self.known_utm) if self.known_utm else None
        return d


@dataclass
class Proposal:
    """A reasoning agent's DIAGNOSIS + one bounded fix to re-attempt.

    The fix is the key of a SAFE_ACTIONS entry (an automation re-run, or a human-input ask).
    A proposal NEVER changes a placement by itself: an AUTO action is re-run through the
    unchanged deterministic gate (``regate``), and ``accepted_by_gate`` records the gate's
    verdict -- so the LLM reasons but the math decides.
    """
    survey_number: str
    hypothesis: str
    action: str                 # a SAFE_ACTIONS key
    rationale: str = ""
    source: str = "rule"        # "rule" (deterministic fallback) or "llm:<provider>"
    is_auto: bool = False       # True if action is a re-runnable automation (vs a human ask)
    regated: bool = False       # whether the re-gate was attempted
    accepted_by_gate: bool = False  # the GATE's verdict after re-running the fix (never the LLM's)
    note: str = ""              # re-gate outcome detail

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentReport:
    """What one agent produced on a job."""
    agent: str
    checks: list[Check] = field(default_factory=list)
    requests: list[InputRequest] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(c.severity == Severity.FAIL for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "passed": not self.failed,
            "checks": [c.to_dict() for c in self.checks],
            "requests": [r.to_dict() for r in self.requests],
            "proposals": [p.to_dict() for p in self.proposals],
            "notes": list(self.notes),
        }


class Agent:
    """Base class: a runtime agent runs on the job's results and returns a report.

    Subclasses override ``run``. They receive the georef results and a small context
    dict (output_dir, crs, cadastral_source, surveyor, ...). They MUST NOT mutate a
    result's recommendation to a confident state -- only the math gates do that.
    """
    name = "agent"

    def run(self, results, context: dict) -> AgentReport:  # noqa: D401
        raise NotImplementedError


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2))
    return path
