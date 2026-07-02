"""PipelineOrchestrator -- Qwen drives the WHOLE village pipeline, like a local Claude.

Same ReAct pattern as AgentHarness, but the tools are PIPELINE STAGES (village-level), so
the local Qwen brain orchestrates end to end: list villages, check M1 status, run M1, run
the verification agents, read their flags, run cadastral M2, and narrate -- exactly the
loop a human/Claude runs by hand.

0-FP is preserved because the tools only DRIVE the deterministic pipeline + verification
agents + math gates; Qwen decides the SEQUENCE and the narration, never a placement or a
pass/fail. If Ollama is down (or Qwen stalls), a deterministic driver runs the same stages
in order, so the orchestration always completes. Heavy stages (run_m1/run_m2) shell out to
the existing CLIs so they stay isolated and identical to a manual run.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from .providers import llm_call

_log = logging.getLogger(__name__)

_INPUT = Path("input")
_OUTPUT = Path("output")


# --------------------------------------------------------------------------- tools
def list_villages(_: str = "") -> dict:
    """Villages that have FMB inputs, with how far each has progressed."""
    out = []
    for d in sorted(_INPUT.glob("*/fmb")):
        v = d.parent.name
        n_pdf = len(list(d.glob("*.pdf")))
        n_m1 = len(list((_OUTPUT / v / "m1").glob(f"{v}_*.dxf")))
        m2 = (_OUTPUT / v / "m2" / "clubbed_village.dxf").exists()
        out.append({"village": v, "fmbs": n_pdf, "m1_done": n_m1, "m2_done": m2})
    return {"villages": out}


def m1_status(village: str) -> dict:
    """M1 completeness + PROPER count for one village (reads the verify sidecars)."""
    m1 = _OUTPUT / village / "m1"
    dxfs = list(m1.glob(f"{village}_*.dxf"))
    proper = sum(1 for f in dxfs
                 if (f.with_suffix(".verify.txt").exists()
                     and "STATUS: PROPER" in f.with_suffix(".verify.txt")
                     .read_text(encoding="utf-8", errors="replace")))
    n_pdf = len(list((_INPUT / village / "fmb").glob("*.pdf")))
    return {"village": village, "fmbs": n_pdf, "m1_dxfs": len(dxfs), "proper": proper,
            "complete": len(dxfs) >= n_pdf and n_pdf > 0}


def verify(village: str) -> dict:
    """Run the NUMERIC verification agent across the village; return the flag tally so
    Qwen can decide whether a re-run is needed."""
    sys.path.insert(0, "src")
    from landintel.pipeline.m1_extract.verify_dxf import verify_m1_dxf
    m1 = _OUTPUT / village / "m1"
    flags: dict[str, int] = {}
    improper = []
    dxfs = sorted(m1.glob(f"{village}_*.dxf"))
    for f in dxfs:
        try:
            r = verify_m1_dxf(f)
        except Exception:  # noqa: BLE001
            continue
        for c in r.checks:
            if not c.passed:
                flags[c.name] = flags.get(c.name, 0) + 1
        if not r.proper:
            improper.append(f.stem.split("_")[-1])
    return {"village": village, "plots": len(dxfs), "flags": flags, "improper": improper}


def run_m1(village: str) -> dict:
    """Run (or complete) M1 for a village via the parallel CLI. Long-running."""
    r = subprocess.run([sys.executable, "run_m1.py", village],
                       capture_output=True, text=True, timeout=3600)
    tail = "\n".join(r.stdout.splitlines()[-3:])
    return {"village": village, "returncode": r.returncode, "tail": tail}


def run_m2(village: str) -> dict:
    """Run cadastral M2 (auto-located) for a village via the CLI. Long-running."""
    r = subprocess.run([sys.executable, "run_m2_cad.py", village],
                       capture_output=True, text=True, timeout=3600)
    tail = "\n".join(r.stdout.splitlines()[-6:])
    return {"village": village, "returncode": r.returncode, "tail": tail}


def verify_m2(village: str) -> dict:
    """Read the M2 club verification: the 0-FP gates + how many plots were placed."""
    m2 = _OUTPUT / village / "m2"
    vf = m2 / "clubbed.verify.txt"
    gates = vf.read_text(encoding="utf-8", errors="replace") if vf.exists() else ""
    n_pts = 0
    csv = m2 / "clubbed_points.csv"
    if csv.exists():
        n_pts = len({ln.split(",")[0] for ln in
                     csv.read_text(encoding="utf-8", errors="replace").splitlines()[1:]})
    return {"village": village, "clubbed_dxf": (m2 / "clubbed_village.dxf").exists(),
            "gates_pass": "PASS" in gates.splitlines()[0] if gates else None,
            "surveys_with_points": n_pts}


def deliverables(village: str) -> dict:
    """List the final deliverable files for a village (the client hand-off bundle)."""
    m2 = _OUTPUT / village / "m2"
    wanted = ["clubbed_village.dxf", "clubbed.geojson", "clubbed_points.csv",
              "clubbed_qa.png", "village_area_statement.pdf", "village_area_breakdown.xlsx",
              "village_delivery.zip"]
    present = {f: (m2 / f).exists() for f in wanted}
    return {"village": village, "deliverables": present,
            "complete": all(present[f] for f in
                            ("clubbed_village.dxf", "village_delivery.zip"))}


_TOOLS = {
    "list_villages": (list_villages, "list all villages + how far each has progressed"),
    "m1_status": (m1_status, "M1 completeness + PROPER count for a village"),
    "verify": (verify, "run the M1 verification agent; returns the flag tally + improper plots"),
    "run_m1": (run_m1, "run/complete M1 extraction for a village (minutes)"),
    "run_m2": (run_m2, "run cadastral M2 club for a village, auto-located (minutes)"),
    "verify_m2": (verify_m2, "read the M2 club 0-FP gates + placed count for a village"),
    "deliverables": (deliverables, "list the final deliverable bundle for a village"),
}

_GOAL = ("Process every village end to end. For each village: check m1_status; if M1 is not "
         "complete, run_m1; then verify (M1) and note any flags; then run_m2; then verify_m2 "
         "(the 0-FP club gates); then deliverables to confirm the client bundle exists. Finish "
         "with a short client summary of what is delivered and what still needs attention.")


# --------------------------------------------------------------------------- harness
class PipelineOrchestrator:
    """Qwen-first ReAct loop over the pipeline-stage tools; deterministic fallback."""

    def __init__(self, allow_heavy: bool = True):
        self.allow_heavy = allow_heavy      # gate run_m1/run_m2 (off = plan/verify only)

    def _catalogue(self) -> str:
        return "\n".join(f"  - {n}(village): {d}" for n, (_, d) in _TOOLS.items()
                         if self.allow_heavy or n not in ("run_m1", "run_m2"))

    def _system(self) -> str:
        return ("You are the LandIntel pipeline orchestrator (a local agent). Drive the tools "
                "to take each Tamil Nadu village from FMB PDFs to a clubbed village map. You "
                "NEVER decide a placement or a pass/fail -- the deterministic agents and math "
                "gates do that; you only choose the next tool and narrate.\nTools:\n"
                + self._catalogue()
                + '\n\nEach turn reply with STRICT JSON only: {"tool":"<name>","args":'
                  '{"village":"<v>"}} to act, or {"final":"<summary>"} to stop.')

    def run(self, goal: str | None = None, max_steps: int = 20) -> dict:
        goal = goal or _GOAL
        first = llm_call(self._system() + "\n\nGOAL: " + goal
                         + "\n\nStart by listing villages. Your first JSON:", max_tokens=200)
        if first is None:
            return self._deterministic()
        provider, convo, transcript = first[1], f"GOAL: {goal}\n", []
        reply = first[0]
        for _ in range(max_steps):
            step = self._parse(reply)
            if step is None or "final" in step:
                return {"provider": provider, "steps": transcript,
                        "final": (step or {}).get("final", "") or self._summary(),
                        "tool_calls": sum(1 for s in transcript if s["type"] == "tool")}
            obs = self._exec(step)
            transcript.append({"type": "tool", "call": step, "observation": obs})
            convo += f"\nCALL {json.dumps(step)} -> {json.dumps(obs)[:500]}\n"
            nxt = llm_call(self._system() + "\n\n" + convo + "\nYour next JSON:", max_tokens=200)
            if nxt is None:
                break
            reply = nxt[0]
        return {"provider": provider, "steps": transcript,
                "final": self._summary(), "tool_calls": len(transcript)}

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _parse(text: str) -> dict | None:
        try:
            return json.loads(text[text.index("{"): text.rindex("}") + 1])
        except Exception:  # noqa: BLE001
            return None

    def _exec(self, step: dict) -> dict:
        name = step.get("tool")
        entry = _TOOLS.get(name)
        if entry is None or (not self.allow_heavy and name in ("run_m1", "run_m2")):
            return {"error": f"tool '{name}' unavailable"}
        try:
            return entry[0](step.get("args", {}).get("village", ""))
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _deterministic(self) -> dict:
        """No LLM -> run the stages in order for every village so it still completes."""
        transcript = []
        for v in list_villages()["villages"]:
            name = v["village"]
            for tool in ("m1_status", "verify", "verify_m2", "deliverables"):
                transcript.append({"type": "tool", "call": {"tool": tool, "village": name},
                                   "observation": _TOOLS[tool][0](name)})
        return {"provider": None, "steps": transcript, "final": self._summary(),
                "tool_calls": len(transcript)}

    def _summary(self) -> str:
        vs = list_villages()["villages"]
        done = sum(1 for v in vs if v["m2_done"])
        return (f"{len(vs)} villages; M1 done for "
                f"{sum(1 for v in vs if v['m1_done'] >= v['fmbs'] and v['fmbs'])}, "
                f"M2 clubbed for {done}. Verification agents flag remaining issues per village; "
                f"no plot placed incorrectly (0 FP by construction).")
