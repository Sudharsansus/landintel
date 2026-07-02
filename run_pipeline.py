"""Qwen orchestrates the LandIntel village pipeline (local agent, like Claude does by hand).

  python run_pipeline.py            # plan + verify only (no heavy runs) -- safe demo
  python run_pipeline.py --run      # let Qwen also run M1/M2 (long-running)

Qwen drives the pipeline-stage tools; the deterministic agents + math gates do the work and
keep it 0-FP; a deterministic driver takes over if Ollama is down.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from landintel.llm.pipeline_agent import PipelineOrchestrator
from landintel.llm.providers import local_llm_status

allow_heavy = "--run" in sys.argv
print("Qwen status:", json.dumps(local_llm_status()))
orch = PipelineOrchestrator(allow_heavy=allow_heavy)
result = orch.run()

print(f"\n=== ORCHESTRATION ({result.get('provider') or 'deterministic fallback'}, "
      f"{result.get('tool_calls')} tool calls) ===")
for s in result.get("steps", []):
    call = s.get("call", {})
    print(f"  -> {call.get('tool', call)}  ::  {json.dumps(s.get('observation'))[:160]}")
print("\nFINAL:", result.get("final"))
