"""Tests for the consolidated LLM brain: memory graph + tool-calling harness.

Invariants under test:
  * the memory graph PERSISTS across sessions and recalls prior dispositions/inputs;
  * the harness drives ONLY FP-safe tools and never places a plot (the gate decides);
  * the harness works fully offline (deterministic driver when no LLM is reachable).
"""
from __future__ import annotations

from landintel.llm.concept import SYSTEM_CONCEPT
from landintel.llm.memory_graph import (
    BASELINE_KNOWLEDGE,
    MemoryGraph,
    seed_baseline_knowledge,
)
from landintel.llm.harness import AgentHarness
from landintel.pipeline.m2_georef.pipeline import GeorefResult


def _r(sn, rec, method="", cov=0.0, resid=float("inf"), m1=None):
    return GeorefResult(
        m1_file=m1 or f"test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_INGUR_{sn}.dxf",
        survey_number=sn, matched=True, recommendation=rec,
        match_method=method, chain_coverage=cov, cad_residual=resid)


# --------------------------------------------------- corrected pipeline concept --
def test_concept_describes_corrected_m1_m2_club_m3_split():
    """The brain's concept must state M1 -> M2(m2_club, FMB-only club) -> M3(m2_georef,
    surveyor assembly), the three M2 methods, and the 0-FP discipline."""
    c = SYSTEM_CONCEPT
    # M2 is the new FMB-only club module (m2_club), described as clubbing WITHOUT a surveyor file
    assert "m2_club" in c
    assert "club" in c.lower()
    # the three cross-checked M2 methods are named
    for method in ("cadastral_seat", "gps_seat", "relative_club"):
        assert method in c, f"M2 method {method!r} missing from concept"
    # m2_georef is now M3 (surveyor assembly), NOT M2
    assert "m2_georef" in c
    assert "M3" in c
    # the old text must not call m2_georef "M2 georef" anymore
    assert "M2 georef" not in c
    # 0-FP discipline and the surveyor raw-data file framing are present
    assert "false positive" in c.lower()
    assert "RAW DATA FILE" in c or "raw-data file" in c


def test_seed_baseline_knowledge_records_corrected_architecture(tmp_path):
    """seed_baseline_knowledge persists the corrected split so the brain recalls it across
    sessions (a fresh graph object on the same file still has it)."""
    path = tmp_path / "mem.json"
    seed_baseline_knowledge(MemoryGraph(path))

    g2 = MemoryGraph(path)                                   # new object, same file
    facts = {f["topic"]: f["fact"] for f in g2.recall_knowledge()}
    assert {"mission", "architecture", "mental_model", "hard_rule"} <= set(facts)
    arch = facts["architecture"]
    assert "m2_club" in arch and "m2_georef" in arch
    assert "M2 club" in arch and "M3 georef" in arch
    # mental model spells out m2_georef is M3, not M2
    assert "m2_club is M2" in facts["mental_model"]
    assert "m2_georef is M3" in facts["mental_model"]
    # idempotent upsert-by-topic: re-seeding does not duplicate the topics
    seed_baseline_knowledge(g2)
    topics = [f["topic"] for f in MemoryGraph(path).recall_knowledge()]
    assert len(topics) == len(set(topics)) == len(BASELINE_KNOWLEDGE)


# ----------------------------------------------------------------- memory graph --
def test_memory_graph_persists_and_recalls_across_sessions(tmp_path):
    path = tmp_path / "mem.json"
    g1 = MemoryGraph(path)
    g1.record_job([_r("668", "REVIEW", "cadastral_rigid_located", 0.30, 40.0),
                   _r("724", "ACCEPT_CADASTRAL", "cadastral_rigid", 1.0, 2.0)],
                  village="INGUR")
    g1.record_operator_input("INGUR", "668", "clearer_parcel", "ACCEPT_SEEDED",
                             "operator gave KML")

    # a NEW graph object reading the SAME file must remember the prior session
    g2 = MemoryGraph(path)
    past = g2.recall("668", village="INGUR")
    assert past["known"] is True
    assert past["resolved_via"] == "clearer_parcel"
    assert {"input_type": "clearer_parcel", "outcome": "ACCEPT_SEEDED"} in past["operator_inputs"]
    assert g2.recall("724", "INGUR")["last_disposition"] == "ACCEPT_CADASTRAL"
    assert g2.stats()["events"] >= 3


def test_knowledge_persists_and_recalls_by_tag(tmp_path):
    """Project knowledge ('what we are doing') the brain carries across sessions."""
    path = tmp_path / "mem.json"
    g1 = MemoryGraph(path)
    g1.record_knowledge("hard_rule", "only math gates accept; LLM proposes only", ["core", "safety"])
    g1.record_knowledge("mission", "FMB -> UTM village DWG, zero false positives", ["core"])

    g2 = MemoryGraph(path)                          # new object, same file
    all_facts = g2.recall_knowledge()
    topics = {f["topic"] for f in all_facts}
    assert {"hard_rule", "mission"} <= topics
    safety = g2.recall_knowledge(tag="safety")
    assert [f["topic"] for f in safety] == ["hard_rule"]   # tag filter works
    assert g2.stats()["by_type"].get("knowledge") == 2


def test_record_knowledge_upserts_by_topic(tmp_path):
    g = MemoryGraph(tmp_path / "mem.json")
    g.record_knowledge("mission", "v1", ["core"])
    g.record_knowledge("mission", "v2 updated", ["core"])     # same topic -> upsert
    facts = g.recall_knowledge()
    assert len(facts) == 1 and facts[0]["fact"] == "v2 updated"


def test_memory_recall_unknown_plot_is_safe(tmp_path):
    g = MemoryGraph(tmp_path / "mem.json")
    past = g.recall("99999", village="INGUR")
    assert past["known"] is False and past["operator_inputs"] == []


# --------------------------------------------------------------------- harness ---
def test_harness_offline_drives_fp_safe_tools_without_placing(monkeypatch, tmp_path):
    monkeypatch.setenv("LANDINTEL_LLM_ORDER", "")        # no LLM -> deterministic driver
    results = [_r("724", "ACCEPT_CADASTRAL", "cadastral_rigid", 1.0, 2.0),
               _r("668", "REVIEW", "cadastral_rigid_located", 0.30, 40.0)]
    before = [r.recommendation for r in results]
    ctx = {"village": "INGUR", "memory_graph": MemoryGraph(tmp_path / "mem.json")}
    out = AgentHarness(results, ctx).run()
    assert out["provider"] is None                       # ran offline
    assert out["tool_calls"] >= 1                        # it actually drove tools
    assert [r.recommendation for r in results] == before  # NOTHING placed by the harness
    # every executed tool was a registered FP-safe tool (no off-vocab tool ran)
    safe = {"diagnose", "recall", "cadastral_identity", "ocr_read", "list_requests",
            "propose_fix"}
    assert all(s["call"]["tool"] in safe for s in out["steps"])


def test_harness_propose_tool_cannot_place_cross_village(monkeypatch, tmp_path):
    monkeypatch.setenv("LANDINTEL_LLM_ORDER", "")
    xv = _r("9", "REVIEW", "cadastral_rigid_located", 0.9, 5.0,
            m1="test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_KANDAMPALAYAM _9.dxf")

    def _stub_accept(r, action):
        r.recommendation = "ACCEPT_CADASTRAL"
        return True, "stub accepted"

    h = AgentHarness([xv], {"village": "INGUR", "reattempt": _stub_accept,
                            "memory_graph": MemoryGraph(tmp_path / "m.json")})
    # even if the harness proposes the auto-fix and the stub gate accepts, the global guard
    # reverts a cross-village plot -> still REVIEW.
    res = h.tools["propose_fix"].fn("9", "road_closure_recover")
    assert res["accepted_by_gate"] is False and xv.recommendation == "REVIEW"
