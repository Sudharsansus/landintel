"""Back-compat shim: the tool surface now lives in ``landintel.llm.tools``.

Kept so existing imports (``landintel.agents.tools``) keep working after the LLM brain was
consolidated under ``landintel.llm``. Import from ``landintel.llm`` in new code.
"""

from __future__ import annotations

from ..llm.tools import Tool, build_tools  # noqa: F401
