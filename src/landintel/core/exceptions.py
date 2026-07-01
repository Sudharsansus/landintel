"""The exception hierarchy for the whole system.

Deliberately shallow. There is one base, one branch per pipeline failure
domain (so the orchestrator and API can react differently per stage), and one
control-flow signal -- :class:`ValidationFlag` -- that is *not* a failure but a
request to stop and route a plot to a human.

We do not enumerate dozens of fine-grained exception types. Catch sites care
about the *domain* that failed (was it OCR? geometry? georeferencing?), not the
exact line, so the branches mirror the modules:

    LandIntelError
    +-- ConfigError          (startup / config misconfiguration)
    +-- OCRFailure           (M1: PaddleOCR could not read the image)
    +-- GeometryError        (M1: vector extraction / DXF geometry is invalid)
    +-- GeoreferenceError    (M2: affine fit or reprojection failed)
    +-- AssemblyError        (M3: village merge / boundary snap failed)
    +-- ReportError          (M4: report generation or delivery failed)
    +-- ValidationFlag       (route to human review; NOT a hard failure)

The distinction that matters most is ``GeometryError`` ("this plot is broken,
fail it") versus ``ValidationFlag`` ("this plot is suspicious, flag it but keep
the village moving"). The anomaly layer raises the latter; the orchestrator
catches it, marks the plot ``FLAGGED``, and continues with the other plots.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LandIntelError",
    "ConfigError",
    "OCRFailure",
    "GeometryError",
    "GeoreferenceError",
    "AssemblyError",
    "ReportError",
    "AgentError",
    "ValidationFlag",
]


class LandIntelError(Exception):
    """Base class for every error raised by LandIntel code.

    Carries an optional ``context`` mapping so the audit trail and structured
    logs can record *which* job / plot / field the failure concerned without
    parsing the message string. Catching :class:`LandIntelError` catches every
    domain failure below it (but intentionally also catches
    :class:`ValidationFlag` -- see its note).
    """

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({details})"


class ConfigError(LandIntelError):
    """Configuration is missing or invalid; raised at startup, fails fast."""


class OCRFailure(LandIntelError):
    """M1: PaddleOCR failed to produce usable text from the FMB image.

    Distinct from a *low-confidence read* (which is a :class:`ValidationFlag`):
    this means OCR could not run or returned nothing at all.
    """


class GeometryError(LandIntelError):
    """M1: vector geometry could not be extracted or is structurally invalid.

    A hard failure for the plot -- e.g. no boundary lines found, or the DXF
    round-trip read-back does not match what was written.
    """


class GeoreferenceError(LandIntelError):
    """M2: the affine transform could not be fit, or reprojection failed.

    Typically a degenerate GPS anchor set (collinear / too few points) or an
    out-of-range coordinate that pyproj rejects.
    """


class AssemblyError(LandIntelError):
    """M3: merging plots into the village map failed unrecoverably.

    Used for structural failures (a plot DXF cannot be merged). A merely
    mismatched shared boundary is a resolver concern, not this error.
    """


class ReportError(LandIntelError):
    """M4: report generation, packaging, or S3 delivery failed."""


class AgentError(LandIntelError):
    """The agent layer could not complete: the LLM call failed after retries, or
    the tool-use loop did not converge. A hard failure, distinct from a
    :class:`ValidationFlag` (which is a normal "route to human" outcome)."""


class ValidationFlag(LandIntelError):
    """Routing signal: stop processing this plot and send it to human review.

    This is **not** a hard failure. It is raised by the agent/anomaly layer when
    a plot is *suspicious but not provably broken* (low-confidence OCR, area
    mismatch within a grey zone, a boundary that almost closes). The
    orchestrator catches it, marks the plot ``FLAGGED`` rather than ``FAILED``,
    records the reason, and continues with the remaining plots so one suspect
    plot never stalls an entire village.

    It subclasses :class:`LandIntelError` so a broad ``except LandIntelError``
    still sees it -- but callers that need the flag/fail distinction must catch
    :class:`ValidationFlag` *before* the broader domain errors.

    Attributes:
        reason: Human-readable explanation shown in the review UI.
        field: The plot field under suspicion (e.g. ``"area"``), if specific.
        severity: Free-form severity tag (e.g. ``"low"``, ``"high"``) the
            dashboard can sort/colour by.
    """

    def __init__(
        self,
        reason: str,
        /,
        *,
        field: str | None = None,
        severity: str = "medium",
        **context: Any,
    ) -> None:
        super().__init__(reason, field=field, severity=severity, **context)
        self.reason = reason
        self.field = field
        self.severity = severity
