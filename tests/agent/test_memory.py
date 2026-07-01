"""Tests for correction memory -- logging only; retrieval stays deferred."""

from __future__ import annotations

import pytest

from landintel.agent.memory import build_correction, log_correction, recall
from landintel.core.enums import MeasurementSource
from landintel.core.models import Correction


class FakeRecorder:
    def __init__(self) -> None:
        self.recorded: list[Correction] = []

    def record(self, correction: Correction) -> None:
        self.recorded.append(correction)


def test_build_correction_is_pure() -> None:
    c = build_correction(
        client_id="cli", job_id="j1", plot_id="100", field="measurement",
        old="98", new="40.6", measurement_ref="123,456",
    )
    assert isinstance(c, Correction)
    assert c.client_id == "cli" and c.job_id == "j1" and c.plot_id == "100"
    assert c.old == "98" and c.new == "40.6"
    assert c.measurement_ref == "123,456"
    assert c.old_source is MeasurementSource.OCR


def test_log_correction_records_and_returns() -> None:
    recorder = FakeRecorder()
    correction = log_correction(
        recorder, client_id="cli", job_id="j1", plot_id="100",
        field="measurement", old="98", new="40.6",
    )
    assert recorder.recorded == [correction]
    assert correction.new == "40.6"


def test_corrections_carry_client_id_for_tenancy() -> None:
    recorder = FakeRecorder()
    log_correction(
        recorder, client_id="tenant_x", job_id="j", plot_id="1",
        field="stated_area", old="1.0", new="1.2",
    )
    assert recorder.recorded[0].client_id == "tenant_x"


def test_recall_is_deferred() -> None:
    """Retrieval must remain an explicit, unbuilt placeholder."""
    with pytest.raises(NotImplementedError):
        recall("anything")
