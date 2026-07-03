"""Optional bridge to opengeos/GeoAgent (Strands-based geospatial agent harness).

GeoAgent is NOT an OCR or maths engine -- it is the agent SHELL: a multi-turn tool-calling
loop with provider support for exactly our combo (Anthropic Claude, Qwen/DeepSeek via
OpenRouter, Ollama, vLLM) and a QGIS chat-UI plugin. This adapter registers LandIntel's
FP-safe tool surface (tools.build_tools) into a GeoAgent so the model can drive OUR OCR engine
and OUR maths/gate engine conversationally -- "another Claude Code wired to our engines."

SAFETY: only our read-only / propose-and-re-gate tools are registered. GeoAgent's
arbitrary-code escape hatch (run_pyqgis_script) is deliberately NOT exposed, and no tool can
set a placement to ACCEPT -- so even a fully autonomous GeoAgent session is 0-FP by
construction (the math gate still decides every placement).

OPTIONAL + FAIL-SAFE (like the Manus provider): if ``geoagent`` is not installed this returns
None and the rest of the pipeline is unaffected. The exact GeoAgent registration API is
feature-detected; this needs a quick validation against the installed GeoAgent version (the
load-bearing, tested part of this feature is tools.py, which is harness-agnostic).
"""

from __future__ import annotations

import logging
import os

from .tools import build_tools

_log = logging.getLogger(__name__)

# map our provider order to GeoAgent's provider names (it also auto-detects from env)
_PROVIDER_MAP = {"local": "ollama", "claude": "anthropic", "manus": "openai"}


def geoagent_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("geoagent") is not None
    except Exception:  # noqa: BLE001
        return False


def build_geoagent(results, context: dict | None = None):
    """Construct a GeoAgent with LandIntel's FP-safe tools registered, or None if GeoAgent
    is not installed / its API differs. Does not change any placement; it only lets an LLM
    call our read/propose tools."""
    if not geoagent_available():
        _log.info("GeoAgent not installed -> bridge skipped (pip install geoagent to enable; "
                  "the FP-safe tool surface in tools.py works with any harness).")
        return None
    try:
        import geoagent as ga
    except Exception as exc:  # noqa: BLE001
        _log.warning("GeoAgent import failed: %s", exc)
        return None

    tools = build_tools(results, context)
    # Strip EVERY item (not just the first) so "local, claude , manus" parses cleanly.
    order = [p.strip() for p in
             os.environ.get("LANDINTEL_LLM_ORDER", "local").split(",")]  # offline-only default
    provider = _PROVIDER_MAP.get(order[0], "anthropic")

    # Feature-detect the registration API (the README documents a @geo_tool decorator + a
    # GeoAgent facade / GeoAgentConfig). Wrap each LandIntel tool; degrade clearly if absent.
    geo_tool = getattr(getattr(ga, "tools", ga), "geo_tool", None) or getattr(ga, "geo_tool", None)
    GeoAgent = getattr(ga, "GeoAgent", None)
    if geo_tool is None or GeoAgent is None:
        _log.warning("Installed GeoAgent API differs (no geo_tool/GeoAgent) -> bridge skipped; "
                     "use tools.build_tools() directly. Validate against this GeoAgent version.")
        return None

    registered = []
    for t in tools:
        try:
            fn = t.fn
            fn.__name__ = t.name
            fn.__doc__ = t.description
            registered.append(geo_tool(fn))            # safety flag stays read/propose-only
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not register tool %s: %s", t.name, exc)

    try:
        cfg = getattr(ga, "GeoAgentConfig", None)
        kwargs = {"tools": registered}
        if cfg is not None:
            kwargs["config"] = cfg(provider=provider)
        agent = GeoAgent(**kwargs)
        _log.info("GeoAgent bridge ready: %d LandIntel tools registered, provider=%s "
                  "(arbitrary-code tools NOT exposed; math gate still decides).",
                  len(registered), provider)
        return agent
    except Exception as exc:  # noqa: BLE001
        _log.warning("GeoAgent construction failed (%s) -> bridge skipped.", exc)
        return None
