"""Corrections repository: append-only logging, scoped by client_id.

This is what ``memory.log_correction`` writes to. No retrieval logic beyond
simple scoped queries — the smart recall (similarity, few-shot) is deferred
until real correction data exists.

``client_id`` is required on every method.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...core.models import Correction

__all__ = ["CorrectionRepository"]


def _to_correction(doc: dict) -> Correction:
    doc.pop("_id", None)
    return Correction.model_validate(doc)


class CorrectionRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["corrections"]

    async def record(self, client_id: str, correction: Correction) -> Correction:
        """Append one correction. ``correction.client_id`` must match."""
        assert correction.client_id == client_id
        await self._col.insert_one(correction.model_dump(mode="json"))
        return correction

    async def list_recent(
        self,
        client_id: str,
        *,
        limit: int = 100,
        plot_id: str | None = None,
    ) -> list[Correction]:
        """Return the most recent corrections for this client."""
        filt: dict = {"client_id": client_id}
        if plot_id is not None:
            filt["plot_id"] = plot_id
        cursor = self._col.find(filt).sort("created_at", -1).limit(limit)
        return [_to_correction(doc) async for doc in cursor]
