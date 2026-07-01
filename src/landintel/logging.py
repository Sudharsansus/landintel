"""Structured JSON logging with first-class job context.

Not a framework -- just enough to get one job's full trace out of production
logs with a single grep:

* :func:`configure_logging` installs a JSON formatter on the root logger (once).
* :func:`bind` / :func:`log_context` attach ``job_id`` / ``plot_id`` (and any
  other fields) to every log line emitted within their scope, via a
  :class:`contextvars.ContextVar` so it is safe across async tasks.
* The formatter automatically merges a
  :class:`~landintel.core.exceptions.LandIntelError`'s ``context`` mapping into
  the record when such an exception is being logged, so the structured detail
  the error carried flows straight into the log without manual unpacking.

Note: this module is ``landintel.logging``; ``import logging`` below resolves to
the standard library (Python 3 absolute imports), not to this module.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from .core.exceptions import LandIntelError

__all__ = ["configure_logging", "bind", "clear_context", "log_context", "get_logger"]

# Per-context bag of fields merged into every record (job_id, plot_id, ...).
_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})

# LogRecord attributes that are intrinsic to the record, not caller-supplied
# extras. Anything on the record outside this set is treated as an extra field.
_RESERVED = frozenset(
    logging.makeLogRecord({}).__dict__.keys()
    | {"message", "asctime", "taskName"}
)


class JSONFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object.

    Emits ``time``/``level``/``logger``/``message`` plus the bound context, any
    caller-supplied ``extra=`` fields, and -- when an exception is attached --
    its type, message, and (for :class:`LandIntelError`) its ``context``.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Bound job/plot context first, so it is always present and greppable.
        payload.update(_context.get())

        # Caller extras passed via logger.info(..., extra={...}).
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            exc = record.exc_info[1]
            payload["error"] = {
                "type": type(exc).__name__ if exc else None,
                "message": str(exc) if exc else None,
            }
            if isinstance(exc, LandIntelError) and exc.context:
                # The structured detail the error carried flows into the log.
                payload["error"]["context"] = exc.context

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str | int = "INFO") -> None:
    """Install the JSON formatter on the root logger. Idempotent.

    Replaces any existing handlers so repeated calls (tests, worker re-init)
    don't stack duplicate output.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def bind(**fields: Any) -> None:
    """Merge ``fields`` into the current logging context for this scope.

    Typically ``bind(job_id=job.id)`` at the start of processing. Prefer
    :func:`log_context` when the binding has a clear begin/end.
    """
    _context.set({**_context.get(), **fields})


def clear_context() -> None:
    """Drop all bound context fields."""
    _context.set({})


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Bind ``fields`` for the duration of the ``with`` block, then restore.

    Safe to nest; the previous context is reset exactly on exit even on error.
    """
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin wrapper for a single import point)."""
    return logging.getLogger(name)


# Type alias documentation aid; not exported.
_ContextMap = Mapping[str, Any]
