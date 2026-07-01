"""Correction memory -- LOGGING ONLY.

Single responsibility right now: capture every human correction as a
:class:`~landintel.core.models.Correction` and hand it to a recorder, from day
one, so that when real correction data exists we can build retrieval on top of
it. Nothing smart is built here.

Deliberately NOT implemented (deferred until there is real data to shape it):
retrieval, few-shot example selection, similarity / vector search, prompt
regeneration. Building any of that now would be guessing at the shape of data we
do not have. :func:`recall` is a clearly-marked placeholder that raises -- do not
fill it in until corrections have accumulated.

Persistence is injected (a :class:`CorrectionRecorder`) so this module stays
decoupled from the database layer; the Mongo-backed recorder lands with the
corrections repository.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..core.enums import MeasurementSource
from ..core.models import Correction

__all__ = ["CorrectionRecorder", "build_correction", "log_correction", "recall"]


class CorrectionRecorder(Protocol):
    """Anything that can persist a correction (the corrections repository)."""

    def record(self, correction: Correction) -> Any:
        """Persist one correction."""
        ...


def build_correction(
    *,
    client_id: str,
    job_id: str,
    plot_id: str,
    field: str,
    old: str,
    new: str,
    measurement_ref: str | None = None,
    old_source: MeasurementSource = MeasurementSource.OCR,
) -> Correction:
    """Construct a :class:`Correction` from a correction event (pure, no I/O)."""
    return Correction(
        client_id=client_id,
        job_id=job_id,
        plot_id=plot_id,
        field=field,
        old=old,
        new=new,
        measurement_ref=measurement_ref,
        old_source=old_source,
    )


def log_correction(recorder: CorrectionRecorder, **fields: Any) -> Correction:
    """Build a correction and hand it to the recorder. Returns the correction.

    This is the whole of correction memory today: write it down. See
    :func:`build_correction` for the accepted fields.
    """
    correction = build_correction(**fields)
    recorder.record(correction)
    return correction


def recall(*_args: Any, **_kwargs: Any) -> Any:
    """DEFERRED. Retrieval of past corrections is intentionally not built.

    We have no correction data yet, so the retrieval shape (similarity, few-shot
    selection, vector search) cannot be designed honestly. Build this only once
    real corrections have accumulated -- not before.
    """
    raise NotImplementedError(
        "correction retrieval is deferred until real correction data exists; "
        "memory.py logs corrections only"
    )
