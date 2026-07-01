"""Celery application — one instance for the process.

Broker and result backend both come from the REDIS_URL environment variable,
read at construction time with a redis:// localhost fallback so tests that
import this module (but never actually connect to Redis) still collect cleanly.

No task definitions live here; tasks import ``celery_app`` from this module
so the app is constructed exactly once regardless of import order.
"""

from __future__ import annotations

import os

from celery import Celery

__all__ = ["celery_app"]


def _make_app() -> Celery:
    # Read REDIS_URL directly (not via get_settings()) so import-time
    # construction works in test environments where ANTHROPIC_API_KEY is unset.
    # Fallback is redis:// (not amqp://) so a misconfigured worker fails clearly
    # at connection time rather than silently routing to the AMQP default.
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # broker= and backend= are constructor args; conf.update() uses different
    # (broker_url/result_backend) names and may not take effect before Celery
    # initialises its connection pool. Constructor is the safe path.
    app = Celery("landintel", broker=redis_url, backend=redis_url,
                 include=["landintel.workers.tasks"])
    app.conf.update(
        task_default_queue="jobs",   # worker starts with --queues=jobs; must match
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = _make_app()
