"""FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..db.client import close_client, get_db
from ..db.indexes import create_indexes
from ..logging import configure_logging
from .routes import files, health, jobs, review


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    # LANDINTEL_SKIP_DB_INIT: skip the index-creation round-trip entirely. Tests set
    # this (the DB dep is overridden per-route via dependency_overrides), so the
    # lifespan no longer blocks ~30s per TestClient on the Mongo server-selection
    # timeout -- the cause of the 14m API suite. Never set in production.
    if os.getenv("LANDINTEL_SKIP_DB_INIT", "") not in ("", "0", "false", "False"):
        yield
        return
    try:
        db = get_db()
        await create_indexes(db)
    except Exception:
        # In test environments the DB dep is overridden at the route level;
        # lifespan index creation is best-effort and must not block test collection.
        pass
    yield
    await close_client()


app = FastAPI(
    title="LandIntel",
    description="FMB to georeferenced DWG, agentically automated.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS_ALLOW_ORIGINS env var: comma-separated list of allowed origins.
# Default covers the Render frontend URL + local Vite dev server.
# Override in Render dashboard if the frontend URL ever changes.
_cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "https://landintel-frontend.onrender.com,http://localhost:5173,http://localhost:3000",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(jobs.router)
app.include_router(files.router)
app.include_router(review.router)
