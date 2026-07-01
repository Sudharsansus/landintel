"""Persistent MEMORY GRAPH -- the pipeline remembers every session.

A small, dependency-free (stdlib JSON) knowledge graph that records what happened on every
job so the LLM brain can RECALL across sessions: "last time survey 668 in INGUR was a merged
parcel; the operator supplied a clearer polygon and it became ACCEPT_SEEDED." This turns the
agent from stateless into one that learns from the whole operating history.

GRAPH MODEL (nodes + typed edges, append-only event log underneath):
  nodes:  village, plot (village/survey), session, proposal, input_request, outcome
  edges:  session-PLACED->plot, plot-HAD->proposal, plot-NEEDS->input_request,
          plot-RESOLVED_AS->outcome, plot-IN->village
Every write also appends to an immutable ``events`` log (nothing is ever forgotten); the node/
edge view is the queryable projection.

SAFETY: memory is advisory only. ``recall`` feeds the LLM context; it can NEVER place a plot
(the math gate still decides every placement), so a stale or wrong memory cannot create a
false positive -- at worst it biases a *proposal*, which is then re-gated.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()


def _default_path() -> Path:
    """Where the graph persists. Override with LANDINTEL_MEMORY_DIR; defaults to a stable
    per-user location so it survives across runs and jobs."""
    base = os.environ.get("LANDINTEL_MEMORY_DIR")
    if base:
        return Path(base) / "memory_graph.json"
    return Path.home() / ".landintel" / "memory_graph.json"


class MemoryGraph:
    """Append-only, JSON-backed graph keyed by (village, survey_number). Thread/process safe
    enough for a single-node worker (file lock-free but atomic-replace writes + an in-proc
    lock); fine for the current single-image deployment."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _default_path()
        self.data = {"nodes": {}, "edges": [], "events": [], "version": 1}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
                self.data.setdefault("events", [])
            except Exception:  # noqa: BLE001 - corrupt file -> start fresh, keep a backup
                try:
                    self.path.rename(self.path.with_suffix(".corrupt"))
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------ writes ----
    def _node(self, node_id: str, ntype: str, **attrs) -> None:
        n = self.data["nodes"].get(node_id, {"id": node_id, "type": ntype})
        n.update(attrs)
        n["type"] = ntype
        self.data["nodes"][node_id] = n

    def _edge(self, src: str, rel: str, dst: str) -> None:
        e = [src, rel, dst]
        if e not in self.data["edges"]:
            self.data["edges"].append(e)

    @staticmethod
    def _plot_id(village: str, sn: str) -> str:
        return f"plot:{village}/{sn}"

    def record_job(self, results, village: str, session_id: str | None = None,
                   proposals=None, requests=None) -> str:
        """Record one georef job: every plot's disposition, the proposals tried (+ gate
        verdicts), and the input requests. Returns the session id. Persists immediately."""
        with _LOCK:
            sid = session_id or f"session:{int(time.time())}"
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._node(sid, "session", started=ts, village=village,
                       n_plots=len(results))
            self._node(f"village:{village}", "village", name=village)

            conf = ("ACCEPT", "ACCEPT_SEEDED", "ACCEPT_CADASTRAL")
            for r in results:
                pid = self._plot_id(village, r.survey_number)
                confident = r.recommendation in conf
                self._node(pid, "plot", village=village, survey_number=r.survey_number,
                           last_disposition=r.recommendation,
                           last_method=getattr(r, "match_method", "") or "",
                           confident=confident, last_session=sid, updated=ts)
                self._edge(pid, "IN", f"village:{village}")
                self._edge(sid, "PLACED" if confident else "REVIEWED", pid)
                self.data["events"].append(
                    {"ts": ts, "session": sid, "kind": "disposition",
                     "plot": pid, "disposition": r.recommendation,
                     "method": getattr(r, "match_method", "") or ""})

            for p in (proposals or []):
                pid = self._plot_id(village, p.survey_number)
                self.data["events"].append(
                    {"ts": ts, "session": sid, "kind": "proposal", "plot": pid,
                     "action": p.action, "hypothesis": p.hypothesis,
                     "accepted_by_gate": p.accepted_by_gate, "source": p.source,
                     "note": p.note})
                self._edge(pid, "HAD_PROPOSAL", sid)

            for q in (requests or []):
                qd = q if isinstance(q, dict) else q.to_dict()
                pid = self._plot_id(village, qd.get("survey_number", "?"))
                self.data["events"].append(
                    {"ts": ts, "session": sid, "kind": "input_request", "plot": pid,
                     "input_type": qd.get("input_type"), "reason": qd.get("reason")})
                self._edge(pid, "NEEDS_INPUT", sid)

            self._save()
            return sid

    def record_m1(self, rows, village: str, session_id: str | None = None) -> str:
        """Record an M1 batch (one village's FMB->DXF run): per-PDF verify outcome (proper /
        which checks failed / stone count). This is how each village TRAINS the M1 triage --
        the LLM later recalls 'this district's plots fail closure when ...'. Returns session id."""
        with _LOCK:
            sid = session_id or f"m1:{village}:{int(time.time())}"
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            ok = sum(1 for r in rows if r.get("ok"))
            proper = sum(1 for r in rows if r.get("proper"))
            self._node(sid, "m1_session", started=ts, village=village,
                       n_pdfs=len(rows), ok=ok, proper=proper)
            self._node(f"village:{village}", "village", name=village)
            for r in rows:
                sn = str(r.get("survey") or Path(r.get("pdf", "?")).stem)
                pid = self._plot_id(village, sn)
                self._node(pid, "plot", village=village, survey_number=sn,
                           m1_proper=bool(r.get("proper")), m1_stones=r.get("stones"),
                           m1_fails=r.get("fails") or [], m1_session=sid, updated=ts)
                self._edge(pid, "IN", f"village:{village}")
                self._edge(sid, "EXTRACTED", pid)
                self.data["events"].append(
                    {"ts": ts, "session": sid, "kind": "m1_verify", "plot": pid,
                     "ok": bool(r.get("ok")), "proper": bool(r.get("proper")),
                     "fails": r.get("fails") or [], "stones": r.get("stones"),
                     "error": r.get("error", "")})
            self._save()
            return sid

    def record_knowledge(self, topic: str, fact: str, tags=None) -> None:
        """Persist a project-level FACT the brain should carry across every session
        ("what we are doing"): architecture, gates, the hard rule, the domain backlog.
        Stored as a 'knowledge' node + event so ``recall_knowledge`` can feed it to the LLM.
        Advisory only -- knowledge never places a plot (the math gate still decides)."""
        with _LOCK:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            nid = f"knowledge:{topic}"
            self._node(nid, "knowledge", topic=topic, fact=fact,
                       tags=list(tags or []), updated=ts)
            self.data["events"].append(
                {"ts": ts, "kind": "knowledge", "topic": topic, "fact": fact})
            self._save()

    def recall_knowledge(self, tag: str | None = None) -> list[dict]:
        """All project knowledge facts (optionally filtered by tag), newest-updated first."""
        ks = [n for n in self.data["nodes"].values() if n.get("type") == "knowledge"]
        if tag is not None:
            ks = [k for k in ks if tag in (k.get("tags") or [])]
        ks.sort(key=lambda k: k.get("updated", ""), reverse=True)
        return [{"topic": k["topic"], "fact": k["fact"], "tags": k.get("tags", [])}
                for k in ks]

    def record_chat(self, sender: str, message: str, reply: str,
                    session_id: str) -> None:
        """Persist one chat turn (operator OR Claude asked; the brain replied) so the
        conversation survives across sessions. Advisory log only -- chat never places a
        plot. Truncated to keep the graph small."""
        with _LOCK:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._node(session_id, "chat_session", started=ts, updated=ts)
            self.data["events"].append(
                {"ts": ts, "kind": "chat", "session": session_id, "sender": sender,
                 "message": (message or "")[:500], "reply": (reply or "")[:1200]})
            self._save()

    def recall_chat(self, session_id: str | None = None, limit: int = 20) -> list[dict]:
        """The most recent chat turns (optionally one session), oldest-first."""
        evs = [e for e in self.data["events"] if e.get("kind") == "chat"
               and (session_id is None or e.get("session") == session_id)]
        return evs[-limit:]

    def record_operator_input(self, village: str, sn: str, input_type: str,
                              outcome: str, detail: str = "") -> None:
        """Record that an operator supplied an input and how the plot resolved (e.g.
        ACCEPT_SEEDED). This is the cross-session LEARNING signal."""
        with _LOCK:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            pid = self._plot_id(village, sn)
            self._node(pid, "plot", village=village, survey_number=sn,
                       last_disposition=outcome, resolved_via=input_type, updated=ts)
            self.data["events"].append(
                {"ts": ts, "kind": "operator_input", "plot": pid,
                 "input_type": input_type, "outcome": outcome, "detail": detail})
            self._save()

    # ------------------------------------------------------------------ reads -----
    def recall(self, sn: str, village: str = "INGUR") -> dict:
        """What do we know about this plot from PAST sessions? Returns the latest known
        disposition + the chronological event history (proposals tried, inputs supplied)."""
        pid = self._plot_id(village, sn)
        node = self.data["nodes"].get(pid)
        events = [e for e in self.data["events"] if e.get("plot") == pid]
        prior_inputs = [e for e in events if e["kind"] == "operator_input"]
        tried = sorted({e["action"] for e in events if e["kind"] == "proposal"})
        return {
            "known": node is not None,
            "last_disposition": (node or {}).get("last_disposition"),
            "resolved_via": (node or {}).get("resolved_via"),
            "actions_tried": tried,
            "operator_inputs": [{"input_type": e.get("input_type"),
                                 "outcome": e.get("outcome")} for e in prior_inputs],
            "n_events": len(events),
        }

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for n in self.data["nodes"].values():
            by_type[n["type"]] = by_type.get(n["type"], 0) + 1
        return {"nodes": len(self.data["nodes"]), "edges": len(self.data["edges"]),
                "events": len(self.data["events"]), "by_type": by_type,
                "path": str(self.path)}

    # ------------------------------------------------------------------ persist ---
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)             # atomic on the same filesystem


# --------------------------------------------------------------- baseline facts ---
# The corrected pipeline architecture (M1 -> M2 m2_club -> M3 m2_georef) the brain must
# carry across EVERY session so chat/coder/teach describe the split correctly. Recorded as
# upsert-by-topic knowledge nodes -- advisory only (knowledge never places a plot). Earlier
# text that called m2_georef "M2" is superseded: m2_club is M2, m2_georef is M3.
BASELINE_KNOWLEDGE = [
    ("mission",
     "LandIntel georeferences Tamil Nadu FMB land-survey plots into one UTM village DWG, "
     "fully automated, with an ABSOLUTE prime directive of ZERO false positives.",
     ["core"]),
    ("architecture",
     "M1 extract (pipeline/m1_extract): FMB PDF -> per-plot FMB DXF in relative metres. "
     "M2 club (pipeline/m2_club, NEW): takes the M1 FMB DXFs ONLY -- NO surveyor raw-data "
     "file -- georeferences each plot to UTM and CLUBS them into ONE georeferenced DXF, using "
     "three cross-checked methods (cadastral_seat survey#->parcel rigid-gated; gps_seat "
     "operator control points; relative_club label-free FMB-to-FMB shared-edge clubbing = the "
     "client's FMBS_STONES_MATCH); club_pipeline(...) -> list[ClubResult]; outputs "
     "clubbed_village.dxf + clubbed.geojson + clubbed_points.csv; dispositions ACCEPT / "
     "ACCEPT_SEEDED / REVIEW / NO_COVERAGE. M3 georef (pipeline/m2_georef, the EXISTING "
     "surveyor-matching code formerly mis-labelled 'M2'): takes M2's clubbed FMBs PLUS the "
     "surveyor RAW DATA FILE.dxf and assembles/matches them via RANSAC stone congruence gated "
     "by chain_coverage on the surveyor SITE DATA LINE; georef_pipeline(...) -> "
     "list[GeorefResult]. M4 report: village DWG + PDF/Excel/zip.",
     ["core", "architecture"]),
    ("mental_model",
     "M1 gives FMB DXF -> M2 (m2_club) georeferences + clubs the FMBs WITHOUT the surveyor "
     "raw-data file -> M3 (m2_georef) takes the clubbed DXF and matches it against the surveyor "
     "raw-data file. m2_club is M2; m2_georef is M3 (NOT M2 -- that label is superseded).",
     ["core", "architecture"]),
    ("hard_rule",
     "Deterministic math gates are the ONLY thing that can ACCEPT a placement. The LLM/brain "
     "may only diagnose, pick one SAFE_ACTION, request input, narrate, or remember -- never "
     "emit geometry, a coordinate, or an accept. Every proposal is re-gated; the brain cannot "
     "create a false positive. 0-FP throughout M1 -> M2 -> M3.",
     ["core", "safety"]),
]


def seed_baseline_knowledge(graph: "MemoryGraph | None" = None) -> "MemoryGraph":
    """Record the corrected M1 -> M2(m2_club) -> M3(m2_georef) architecture into the graph so
    the persistent brain recalls it across sessions. Idempotent (upsert by topic). Returns the
    graph. Safe to call any time -- knowledge is advisory only; the math gate still decides."""
    g = graph if graph is not None else default_graph()
    for topic, fact, tags in BASELINE_KNOWLEDGE:
        g.record_knowledge(topic, fact, tags)
    return g


_DEFAULT: MemoryGraph | None = None


def default_graph() -> MemoryGraph:
    """Process-wide singleton graph at the default path."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MemoryGraph()
    return _DEFAULT
