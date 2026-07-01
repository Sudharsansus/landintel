"""Back-compat shim: the concept now lives in ``landintel.llm.concept``.

Kept so existing imports (``landintel.agents.concept``) keep working after the LLM brain was
consolidated under ``landintel.llm``. Import from ``landintel.llm`` in new code.
"""

from __future__ import annotations

from ..llm.concept import (  # noqa: F401
    AUTO_ACTIONS, INPUT_ACTIONS, SAFE_ACTIONS, SYSTEM_CONCEPT,
    parse_proposal, plot_evidence, propose_prompt, rule_based_proposal,
)
