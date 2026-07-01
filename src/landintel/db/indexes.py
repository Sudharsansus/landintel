"""Create the compound indexes needed for tenant-scoped queries.

Called once on startup (from the FastAPI lifespan). All indexes lead with
``client_id`` because every query filters by it first. ``create_index`` is
idempotent: re-running on an already-indexed collection is a no-op.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING

__all__ = ["create_indexes"]


async def create_indexes(db: AsyncIOMotorDatabase) -> None:
    """Ensure all required indexes exist. Safe to call on every startup."""

    # jobs — dashboard "my recent jobs" view (status filter + time sort)
    await db["jobs"].create_index(
        [("client_id", ASCENDING), ("status", ASCENDING)],
        name="jobs_client_status",
        background=True,
    )
    await db["jobs"].create_index(
        [("client_id", ASCENDING), ("created_at", ASCENDING)],
        name="jobs_client_created_at",
        background=True,
    )

    # plots — look up a survey number within a tenant
    await db["plots"].create_index(
        [("client_id", ASCENDING), ("survey_no", ASCENDING)],
        name="plots_client_survey_no",
        background=True,
    )

    # corrections — chronological feed per tenant (append-only)
    await db["corrections"].create_index(
        [("client_id", ASCENDING), ("created_at", ASCENDING)],
        name="corrections_client_created_at",
        background=True,
    )
