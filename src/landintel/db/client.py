"""Motor async Mongo client — one instance for the process.

The use-time ConfigError pattern: a missing or blank MONGO_URI is not checked at
import or startup. It raises :class:`~landintel.core.exceptions.ConfigError`
naming the missing key exactly when :func:`get_db` is first called — which is
when the db subsystem actually activates. This mirrors the S3/ODA pattern and
means the app can boot (and M1/agent tests run) without a live Mongo.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from ..config import get_settings
from ..core.exceptions import ConfigError

__all__ = ["get_db", "close_client"]

_client: AsyncIOMotorClient | None = None


def get_db() -> AsyncIOMotorDatabase:
    """Return the process-wide motor database, creating the client on first call.

    Raises:
        ConfigError: If ``MONGO_URI`` is blank -- named explicitly so the
            operator knows which key to set, rather than getting a cryptic
            motor connection error.
    """
    global _client
    settings = get_settings()
    if not settings.mongo_uri:
        raise ConfigError(
            "MONGO_URI is required to use the database layer; "
            "set it in .env or the environment.",
            missing="mongo_uri",
        )
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client[settings.mongo_db]


async def close_client() -> None:
    """Close the motor client on shutdown (call from the FastAPI lifespan)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
