"""Global pytest configuration.

Set the test-only environment BEFORE any application module is imported, so the
FastAPI lifespan skips its Mongo index-creation round-trip. Without this each
``with TestClient(app)`` blocks ~30 s on the Mongo server-selection timeout (the
DB dep is overridden per-route in tests anyway) -- the cause of the multi-minute
API suite. Setting it at conftest import time (root conftest, collected first)
guarantees it is in place before any TestClient lifespan fires.
"""

from __future__ import annotations

import os

os.environ.setdefault("LANDINTEL_SKIP_DB_INIT", "1")
