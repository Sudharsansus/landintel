"""FastAPI dependencies — injected into every route.

``current_client()`` is the tenant gate. Every route that reads or writes data
declares it as a dependency, so tenant-scoping is structural: there is no route
that can accidentally skip it because the dependency signature makes it explicit.

Current behaviour: returns the single hardcoded client from config.
Future behaviour: reads the Bearer token, validates it, returns the tenant it
identifies. Swapping the implementation here propagates to every route
automatically — no per-route changes needed.
"""

from __future__ import annotations

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..config import get_settings
from ..db.client import get_db as _get_db
from ..db.repositories.corrections import CorrectionRepository
from ..db.repositories.jobs import JobRepository
from ..db.repositories.plots import PlotRepository

__all__ = [
    "current_client",
    "get_database",
    "get_job_repo",
    "get_plot_repo",
    "get_correction_repo",
]


def current_client() -> str:
    """Return the client_id for the current request.

    Today: the single hardcoded client from config.
    Later: validate the Bearer token and return the tenant it identifies.
    Every route that accesses data must declare this dependency.
    """
    return get_settings().client_id


def get_database() -> AsyncIOMotorDatabase:
    """Return the motor database (process-wide singleton)."""
    return _get_db()


def get_job_repo(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> JobRepository:
    return JobRepository(db)


def get_plot_repo(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> PlotRepository:
    return PlotRepository(db)


def get_correction_repo(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> CorrectionRepository:
    return CorrectionRepository(db)
