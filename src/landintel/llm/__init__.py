"""landintel.llm -- the LLM BRAIN of the pipeline (separated per product design).

This package is the consolidated "agentic" subsystem: a LOCAL-FIRST LLM engine (Qwen via
Ollama, escalating to Claude, then Manus), our whole-pipeline CONCEPT loaded into the model,
the FP-safe TOOL surface over the OCR + maths engines, a tool-calling HARNESS (knowledge
distilled from opengeos/GeoAgent, fine-tuned to our concept), and a persistent MEMORY GRAPH
that remembers every session.

THE INVARIANT (unchanged): the LLM brain only reads, proposes, narrates, and remembers. It
NEVER decides a placement -- the deterministic math gates do. So no model and no memory recall
can create a false positive.
"""

from __future__ import annotations

from .concept import SAFE_ACTIONS, SYSTEM_CONCEPT, rule_based_proposal
from .memory_graph import (
    BASELINE_KNOWLEDGE,
    MemoryGraph,
    default_graph,
    seed_baseline_knowledge,
)
from .providers import llm_call
from .tools import Tool, build_tools

__all__ = [
    "llm_call", "SYSTEM_CONCEPT", "SAFE_ACTIONS", "rule_based_proposal",
    "build_tools", "Tool", "MemoryGraph", "default_graph",
    "seed_baseline_knowledge", "BASELINE_KNOWLEDGE",
]
