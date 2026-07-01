"""Plot repository: data access only, no business logic.

``client_id`` is required on every method. Returns domain
:class:`~landintel.core.models.Plot` objects.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...core.enums import PlotStatus
from ...core.models import Plot

__all__ = ["PlotRepository"]


def _to_plot(doc: dict) -> Plot:
    doc.pop("_id", None)
    return Plot.model_validate(doc)


class PlotRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["plots"]

    async def upsert(self, client_id: str, plot: Plot) -> Plot:
        """Insert or replace the plot (keyed by client_id + survey_no)."""
        assert plot.client_id == client_id
        doc = plot.model_dump(mode="json")
        await self._col.replace_one(
            {"client_id": client_id, "survey_no": plot.survey_no},
            doc,
            upsert=True,
        )
        return plot

    async def get(self, client_id: str, survey_no: str) -> Plot | None:
        doc = await self._col.find_one({"client_id": client_id, "survey_no": survey_no})
        return _to_plot(doc) if doc else None

    async def list_by_status(
        self, client_id: str, status: PlotStatus, *, limit: int = 100
    ) -> list[Plot]:
        cursor = self._col.find(
            {"client_id": client_id, "status": status.value}
        ).limit(limit)
        return [_to_plot(doc) async for doc in cursor]

    async def update_status(
        self, client_id: str, survey_no: str, status: PlotStatus
    ) -> bool:
        result = await self._col.update_one(
            {"client_id": client_id, "survey_no": survey_no},
            {"$set": {"status": status.value}},
        )
        return result.matched_count == 1
