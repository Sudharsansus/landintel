# `landintel.llm` — the LLM brain

The consolidated **agentic subsystem**: a local-first LLM engine, the whole pipeline concept
loaded into the model, an FP-safe tool surface over the OCR + maths engines, a tool-calling
harness (knowledge distilled from [opengeos/GeoAgent](https://github.com/opengeos/GeoAgent),
fine-tuned to our concept), and a persistent memory graph that **remembers every session**.

> **The one invariant:** the LLM brain only **reads, proposes, narrates, and remembers** — it
> **never decides a placement**. The deterministic math gates do. So no model and no memory
> recall can ever create a false positive.

## Modules

| File | Role |
|---|---|
| `providers.py` | LLM engine — **Qwen-first** chain: local Qwen (Ollama/vLLM) → Claude → Manus, fail-safe. `llm_call(prompt, system=…)`, `local_llm_status()`. |
| `concept.py` | `SYSTEM_CONCEPT` — the entire pipeline / gates / failure modes loaded into the model; `SAFE_ACTIONS` whitelist; `rule_based_proposal` offline diagnoser. |
| `tools.py` | FP-safe **tool surface** over the real engines: `diagnose`, `cadastral_identity`, `ocr_read`, `recall`, `list_requests` (read-only) and `propose_fix` (propose→re-gate). No tool can place a plot. |
| `harness.py` | `AgentHarness` — a ReAct tool-calling loop (GeoAgent's pattern), Qwen-powered, with a deterministic offline driver. |
| `memory_graph.py` | `MemoryGraph` — persistent JSON knowledge graph; records every job's plots/proposals/inputs and `recall()`s them across sessions. |
| `setup_qwen.ps1` | One-command local Qwen install (Ollama + `qwen2.5:7b`). |

## Run Qwen locally

```powershell
powershell -ExecutionPolicy Bypass -File src/landintel/llm/setup_qwen.ps1
# then verify:
python -c "from landintel.llm.providers import local_llm_status; print(local_llm_status())"
```

Env: `LANDINTEL_LLM_ORDER` (default `local` = OFFLINE-ONLY; set `local,claude` to opt into a
cloud fallback), `LANDINTEL_LOCAL_LLM` (default `qwen2.5:7b`), `OLLAMA_HOST`
(default `http://localhost:11434`), `LANDINTEL_MEMORY_DIR` (memory graph; default `~/.landintel/`).

Deployment: local-host, fully offline. The brain runs entirely on local Qwen by default and
never reaches the internet unless you explicitly add a cloud provider to `LANDINTEL_LLM_ORDER`.

## Why GeoAgent is *referenced*, not *depended on*

GeoAgent is an agent **harness** (AWS Strands), not an OCR/maths engine. We took the useful
idea — a tool-calling loop with our exact provider combo — and reimplemented it dependency-free
and fine-tuned to our concept (`harness.py`). The optional `agents/geoagent_adapter.py` can
still register our tools into a real GeoAgent (e.g. for its QGIS chat UI); the arbitrary-code
escape hatch is deliberately not exposed. Either way the math gate decides every placement.
