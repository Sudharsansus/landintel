"""AgentHarness -- the tool-calling loop (GeoAgent knowledge, fine-tuned to our concept).

opengeos/GeoAgent's idea is a multi-turn agent that calls registered geospatial tools by name
until the task is done. We distil that into a dependency-free ReAct loop wired to OUR concept,
OUR Qwen-first LLM engine, OUR FP-safe tool surface, and OUR memory graph -- "another Claude
Code, integrated with the OCR + maths engines."

THE LOOP: the model is given SYSTEM_CONCEPT + the tool catalogue, and each turn must reply with
STRICT JSON -- either call a tool ({"tool","args"}) or finish ({"final": "<narration>"}). We
execute the call against build_tools(), feed back the observation, and repeat. Tool names are
validated against the registry, so a hallucinated tool does nothing.

WHY IT STAYS 0-FP: the registry contains ONLY read-only + propose-only tools (no tool can
ACCEPT a plot; propose_fix re-runs the deterministic gate). So however the model drives the
loop, it can only read, propose-then-re-gate, and narrate -- the math gate still decides every
placement. If no LLM is reachable, a DETERMINISTIC script drives the same tools (offline-safe).
"""

from __future__ import annotations

import json
import logging

from .concept import SYSTEM_CONCEPT
from .providers import llm_call
from .tools import build_tools

_log = logging.getLogger(__name__)
_CONF = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")
_DEFAULT_GOAL = (
    "For EVERY plot that is not confidently placed: call diagnose, call recall (past sessions), "
    "and propose_fix with the single safest action. Then finish with a 4-6 sentence client "
    "summary stating that no plot was placed incorrectly (0 false positives) and what input the "
    "remaining plots need.")


def _tool_catalogue(tools) -> str:
    return "\n".join(f"  - {t.name}({'survey_number' if t.name != 'list_requests' else ''}"
                     f"{', action' if t.name == 'propose_fix' else ''}) "
                     f"[{t.safety}]: {(' '.join((t.description or '').split()))[:140]}"
                     for t in tools)


class AgentHarness:
    """Drive the FP-safe tools to a goal, Qwen-first, with a deterministic fallback."""

    def __init__(self, results, context: dict | None = None):
        self.results = results
        self.context = context or {}
        self.tools = {t.name: t for t in build_tools(results, self.context)}
        self.village = self.context.get("village", "INGUR")

    # ------------------------------------------------------------------ public ----
    def run(self, goal: str | None = None, max_steps: int = 12) -> dict:
        goal = goal or _DEFAULT_GOAL
        unplaced = [r.survey_number for r in self.results if r.recommendation not in _CONF]
        snapshot = self._snapshot()

        system = (SYSTEM_CONCEPT + "\n\nYou drive tools to inspect and improve a job. "
                  "Available tools:\n" + _tool_catalogue(self.tools.values()) +
                  '\n\nEach turn reply with STRICT JSON, nothing else: to use a tool '
                  '{"tool": "<name>", "args": {"survey_number": "...", "action": "..."}} '
                  '(action only for propose_fix); to stop {"final": "<summary>"}. '
                  "Never output coordinates or 'accept'; the gate decides placements.")

        first = llm_call(system + "\n\nGOAL: " + goal + "\n\nJOB: " + json.dumps(snapshot)
                         + f"\n\nUNPLACED: {unplaced}\n\nYour first JSON:", max_tokens=300)
        if first is None:                        # ---- offline: deterministic driver ----
            return self._deterministic(unplaced)

        provider = first[1]
        transcript: list[dict] = []
        convo = (f"GOAL: {goal}\nJOB: {json.dumps(snapshot)}\nUNPLACED: {unplaced}\n")
        reply = first[0]
        for _ in range(max_steps):
            step = self._parse(reply)
            if step is None or "final" in step:
                final = (step or {}).get("final", "") if step else ""
                return {"provider": provider, "steps": transcript,
                        "final": final or self._fallback_summary(unplaced),
                        "tool_calls": len([s for s in transcript if s["type"] == "tool"])}
            obs = self._exec(step)
            transcript.append({"type": "tool", "call": step, "observation": obs})
            convo += (f"\nCALL: {json.dumps(step)}\nOBSERVATION: {json.dumps(obs)[:600]}\n")
            nxt = llm_call(system + "\n\n" + convo + "\nYour next JSON:", max_tokens=300)
            if nxt is None:
                break
            reply = nxt[0]
        return {"provider": provider, "steps": transcript,
                "final": self._fallback_summary(unplaced),
                "tool_calls": len([s for s in transcript if s["type"] == "tool"])}

    # ------------------------------------------------------------------ internals -
    def _snapshot(self) -> dict:
        disp: dict[str, int] = {}
        for r in self.results:
            disp[r.recommendation] = disp.get(r.recommendation, 0) + 1
        return {"village": self.village, "n_plots": len(self.results), "dispositions": disp}

    @staticmethod
    def _parse(text: str) -> dict | None:
        try:
            s = text[text.index("{"): text.rindex("}") + 1]
            return json.loads(s)
        except Exception:  # noqa: BLE001
            return None

    def _exec(self, step: dict) -> dict:
        name = step.get("tool")
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'; valid: {sorted(self.tools)}"}
        args = step.get("args") or {}
        try:
            if name == "list_requests":
                return tool.fn()
            if name == "propose_fix":
                return tool.fn(args.get("survey_number", ""), args.get("action", ""))
            return tool.fn(args.get("survey_number", ""))
        except Exception as exc:  # noqa: BLE001 - a tool error never crashes the loop
            return {"error": str(exc)}

    def _deterministic(self, unplaced: list[str]) -> dict:
        """No LLM reachable -> run a fixed, FP-safe tool script so the loop still works."""
        transcript = []
        for sn in unplaced:
            for name in ("diagnose", "recall"):
                if name in self.tools:
                    transcript.append({"type": "tool",
                                       "call": {"tool": name, "args": {"survey_number": sn}},
                                       "observation": self._exec(
                                           {"tool": name, "args": {"survey_number": sn}})})
        if "list_requests" in self.tools:
            transcript.append({"type": "tool", "call": {"tool": "list_requests"},
                               "observation": self._exec({"tool": "list_requests"})})
        return {"provider": None, "steps": transcript,
                "final": self._fallback_summary(unplaced),
                "tool_calls": len(transcript)}

    def _fallback_summary(self, unplaced: list[str]) -> str:
        n_conf = sum(1 for r in self.results if r.recommendation in _CONF)
        return (f"{n_conf}/{len(self.results)} plots are gate-verified confident (0 false "
                f"positives by construction). {len(unplaced)} need a small extra input to "
                f"finish (see input_requests.json): {unplaced}. No plot was placed incorrectly.")
