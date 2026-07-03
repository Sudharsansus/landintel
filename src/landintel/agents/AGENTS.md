# LandIntel Agents — single inventory

The one place to see EVERY agent in the system. Golden rule (see top of CLAUDE.md): agents only
**measure / verify / request input / narrate / propose** — a placement's ACCEPT is decided solely
by the deterministic math gates, so **no agent can ever produce a false positive**.

## A. Runtime agents — `src/landintel/agents/` (this folder)
| Agent | File | One job |
|---|---|---|
| `Agent` (base) | `base.py` | Base class + `AgentReport` / `Check` / `InputRequest`. |
| `VerificationAgent` | `verification.py` | Asserts FP-safety + per-module invariants; FAIL → don't ship. |
| `GuardAgent` | `guard.py` | Deterministic final-line FP guards (e.g. 180° anagram trap). |
| `InputRequestAgent` | `input_request.py` | The MINIMAL extra input that closes each unplaced plot → 100%. |
| `LLMAssistAgent` | `llm_assist.py` | Local Qwen + Claude combo for fuzzy tasks; never decides. |
| **`CoordinateFinderAgent`** | `coordinate_finder.py` | **NEW.** Finds a village's EXACT lat/lon automatically: rough web geocode → refine to the TNGIS cadastre village centroid by survey number. Replaces manual `--lat/--lon`. |
| support | `concept.py`, `dispositions.py`, `geoagent_adapter.py`, `geojson.py`, `orchestrator.py`, `regate.py`, `tools.py` | shared helpers / the re-gate + disposition glue. |

## B. Pipeline (M2-club) agents — `src/landintel/agents/club_agents.py` (moved here 2026-07-03)
| Agent | One job |
|---|---|
| `ParcelAgent` | Located cadastre → clean `{survey → parcel polygon}`; flags sliver/merged parcels. |
| `TngisOverlayAgent` | Measures IoU of each placed FMB vs its own cadastre parcel + plot-vs-plot overlap. |
| `AssemblyAgent` | Keeps the club a coherent tiling — demotes (never promotes) overlapping ACCEPTs. |
| `overlay_gate` (fn) | The single 0-FP disposition arbiter (strong / seated / same-parcel-containment). |

## C. Claude agent layer — `src/landintel/agent/` (singular)
Function-style agents Claude orchestrates: `validator.py` (OCR sanity), `anomaly.py`
(closure / area-vs-stated), `resolver.py` (shared-boundary conflicts), `audit.py` (NL audit
trail), `memory.py` (corrections log).

## D. LLM brain — `src/landintel/llm/`
`CodingAgent` (`coder.py`), `PipelineOrchestrator` (`pipeline_agent.py`), plus `harness.py`,
`providers.py` (Qwen-first), `memory_graph.py`, `chat.py`, `concept.py`, `tools.py`.

## Count
**11 agent classes** (6 runtime incl. the new CoordinateFinderAgent + 3 M2-club + 2 LLM) and
the **5 Claude-layer function agents** (validator/anomaly/resolver/audit/memory).

## Consolidation note
Agents currently live in 4 places (A–D). Folders `agent/` (singular) and `agents/` (plural) are
easy to confuse and both define `concept.py`/`tools.py`, so a physical merge is a careful refactor
(updates imports in `llm/tools.py`, `pipeline/orchestrator.py`, `run_m2_cad.py`, and ~10 test
files, and resolves the name collisions) — tracked as the next step. This file is the single
index to review them from until then.
