"""Tests for the runtime agent layer (verification / guard / input-request).

The invariant under test: agents only verify, request input, or narrate -- they never
promote a plot to ACCEPT. So these confirm the FP-safety checks fire and the path-to-100%
worklist maps each unplaced plot to the right minimal input.
"""
from __future__ import annotations

from landintel.agents.guard import rot180
from landintel.agents.input_request import InputRequestAgent
from landintel.agents.llm_assist import LLMAssistAgent
from landintel.agents.verification import VerificationAgent
from landintel.agents.base import InputType, Proposal, Severity
from landintel.agents.concept import (AUTO_ACTIONS, SAFE_ACTIONS, parse_proposal,
                                      rule_based_proposal)
from landintel.agents.regate import regate_proposals
from landintel.pipeline.m2_georef.pipeline import GeorefResult


def _r(sn, rec, method="", cov=0.0, resid=float("inf"), m1=None):
    return GeorefResult(
        m1_file=m1 or f"test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_INGUR_{sn}.dxf",
        survey_number=sn, matched=True, recommendation=rec,
        match_method=method, chain_coverage=cov, cad_residual=resid)


# ---------------------------------------------------------------- rot180 -------
def test_rot180_anagram():
    assert rot180("669") == "699"     # the INGUR phantom
    assert rot180("669") != "669"
    assert rot180("123") is None      # 2,3 don't rotate
    assert rot180("88") == "88"


# ------------------------------------------------ InputRequestAgent (path to 100) --
def test_input_request_maps_minimal_input():
    results = [
        _r("763", "ACCEPT_CADASTRAL", "cadastral_rigid", 1.0, 2.0),   # confident -> no request
        _r("668", "REVIEW", "cadastral_rigid_located", 0.78, 19.0),   # bad parcel -> clearer parcel
        _r("699", "REVIEW", "geometric_6/7_inliers", 0.25),           # partial trace -> confirm
        _r("9", "NO_COVERAGE", "",
           m1="test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_KANDAMPALAYAM _9.dxf"),  # cross-village
    ]
    rep = InputRequestAgent().run(results, {"village": "INGUR"})
    by = {q.survey_number: q.input_type for q in rep.requests}
    assert "763" not in by                                   # confident plots get no request
    assert by["668"] == InputType.CLEARER_PARCEL
    assert by["699"] == InputType.CONFIRM_PLACEMENT
    assert by["9"] == InputType.VILLAGE_REFERENCE
    assert len(rep.requests) == 3                            # one per non-confident plot


# ------------------------------------------------------- VerificationAgent ------
def test_verification_passes_clean_job():
    # No output files -> no footprints -> overlap/verify checks trivially pass; the
    # coverage-accounting + cross-village checks still run.
    results = [_r("763", "ACCEPT_CADASTRAL", "cadastral_rigid", 1.0, 2.0),
               _r("668", "REVIEW", "cadastral_rigid_located", 0.78, 19.0)]
    rep = VerificationAgent().run(results, {"village": "INGUR"})
    assert not rep.failed
    names = {c.name: c.severity for c in rep.checks}
    assert names["coverage_accounting"] == Severity.OK
    assert names["fp_no_cross_village_in_confident"] == Severity.OK


def test_review_geojson_wgs84_and_attributes(tmp_path):
    # A review plot with no DXF footprint but a known UTM point -> a WGS84 Point feature
    # carrying the input-request attributes, reprojected into the INGUR/Erode region.
    import json
    from landintel.agents.geojson import write_review_geojson
    r = _r("668", "REVIEW", "cadastral_rigid_located", 0.78, 19.0)
    reqs = {"668": {"survey_number": "668", "known_utm": [782471.0, 1240376.0],
                    "input_type": "clearer_parcel", "reason": "merged parcel",
                    "instruction": "provide clearer polygon or 2 corners"}}
    path = write_review_geojson([r], reqs, tmp_path, "EPSG:32643")
    fc = json.loads(path.read_text())
    assert len(fc["features"]) == 1
    f = fc["features"][0]
    assert f["geometry"]["type"] == "Point"
    lon, lat = f["geometry"]["coordinates"]
    assert 77.4 < lon < 77.8 and 11.0 < lat < 11.4         # Erode/INGUR region
    assert f["properties"]["status"] == "review"
    assert f["properties"]["input_type"] == "clearer_parcel"


# ---------------------------------------------- reasoning: propose -> re-gate --
def test_rule_based_proposal_maps_failure_modes():
    # the offline diagnoser (no LLM) must map each known failure mode to a SAFE action
    merged = rule_based_proposal(
        {"match_method": "cadastral_rigid_located", "disposition": "REVIEW",
         "area_ratio": 0.30, "cad_residual_m": 40.0}, cross_village=False)
    assert merged["action"] == "road_closure_recover"      # an AUTO automation
    assert merged["action"] in AUTO_ACTIONS
    xv = rule_based_proposal({"match_method": "", "disposition": "NO_COVERAGE"},
                             cross_village=True)
    assert xv["action"] == "request_village_reference"     # a human-input ask


def test_parse_proposal_clamps_offvocab_action():
    ev = {"survey_number": "668"}
    # a hallucinated action that is not in SAFE_ACTIONS must be dropped -> None
    assert parse_proposal('{"action": "move_it_left_5m", "hypothesis": "x"}', ev) is None
    good = parse_proposal('{"action": "road_closure_recover", "hypothesis": "merged"}', ev)
    assert good is not None and good["action"] in SAFE_ACTIONS


def test_reasoning_emits_bounded_proposals_and_never_places(monkeypatch):
    # No LLM reachable -> rule-based proposals. Every proposal's action is in SAFE_ACTIONS,
    # and running the agent must NOT change any recommendation (LLM never decides).
    monkeypatch.setenv("LANDINTEL_LLM_ORDER", "")          # disable all providers -> fast
    results = [_r("763", "ACCEPT_CADASTRAL", "cadastral_rigid", 1.0, 2.0),
               _r("668", "REVIEW", "cadastral_rigid_located", 0.30, 40.0),
               _r("699", "REVIEW", "geometric_6/7_inliers", 0.25)]
    before = [r.recommendation for r in results]
    rep = LLMAssistAgent().run(results, {"village": "INGUR"})
    assert [r.recommendation for r in results] == before   # nothing placed by the agent
    assert {p.survey_number for p in rep.proposals} == {"668", "699"}  # only unplaced
    assert all(p.action in SAFE_ACTIONS for p in rep.proposals)


def test_regate_global_fp_guard_reverts_cross_village(monkeypatch):
    # A re-attempt hook that "accepts" a cross-village plot must be REVERTED by the global
    # FP guard -- 0 false positives outranks recall, even when the (stub) gate said yes.
    xv = _r("9", "REVIEW", "cadastral_rigid_located", 0.9, 5.0,
            m1="test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_KANDAMPALAYAM _9.dxf")
    props = [Proposal("9", "merged", "road_closure_recover", is_auto=True)]

    def _stub_accept(r, action):
        r.recommendation = "ACCEPT_CADASTRAL"             # the stub gate "accepts"
        return True, "stub accepted"

    summary = regate_proposals([xv], props,
                               {"village": "INGUR", "reattempt": _stub_accept})
    assert xv.recommendation == "REVIEW"                   # reverted: cross-village guard
    assert props[0].accepted_by_gate is False
    assert summary["accepted"] == 0


def test_regate_accepts_clean_fix_via_gate():
    # A non-cross-village plot the (stub) gate accepts, with no footprint to overlap, sticks.
    ok = _r("724", "REVIEW", "cadastral_rigid_located", 0.9, 5.0)
    props = [Proposal("724", "merged", "road_closure_recover", is_auto=True)]

    def _stub_accept(r, action):
        r.recommendation = "ACCEPT_CADASTRAL"
        return True, "stub accepted"

    summary = regate_proposals([ok], props,
                               {"village": "INGUR", "reattempt": _stub_accept})
    assert ok.recommendation == "ACCEPT_CADASTRAL"         # gate verdict stuck
    assert props[0].accepted_by_gate is True and summary["accepted"] == 1


def test_regate_defers_when_no_hook():
    # Without a re-attempt hook the AUTO proposal is honestly DEFERRED, not accepted.
    r = _r("668", "REVIEW", "cadastral_rigid_located", 0.30, 40.0)
    props = [Proposal("668", "merged", "road_closure_recover", is_auto=True)]
    summary = regate_proposals([r], props, {"village": "INGUR"})
    assert r.recommendation == "REVIEW"
    assert props[0].regated is False and summary["deferred"] == 1


# ---------------------------------------- FP-safe tool surface (any harness) ----
def test_tools_are_read_or_propose_only():
    # the tool surface any harness (GeoAgent / Claude Code) drives must expose NO tool
    # that can place a plot -- only read_only + proposes.
    from landintel.agents.tools import build_tools
    tools = build_tools([_r("724", "REVIEW", "cadastral_rigid_located", 0.3, 40.0)], {})
    assert {t.safety for t in tools} <= {"read_only", "proposes"}
    assert "propose_fix" in {t.name for t in tools}


def test_tool_propose_fix_rejects_offvocab_and_defers_without_hook():
    from landintel.agents.tools import build_tools
    r = _r("724", "REVIEW", "cadastral_rigid_located", 0.30, 40.0)
    tools = {t.name: t for t in build_tools([r], {"village": "INGUR"})}
    bad = tools["propose_fix"].fn("724", "teleport_it")     # off-vocab -> refused
    assert bad["ok"] is False
    ok = tools["propose_fix"].fn("724", "road_closure_recover")  # no hook -> deferred
    assert ok["accepted_by_gate"] is False and r.recommendation == "REVIEW"


def test_tool_propose_fix_cannot_create_cross_village_fp():
    # Even if a (stub) gate accepts, the global FP guard reverts a cross-village plot.
    from landintel.agents.tools import build_tools
    xv = _r("9", "REVIEW", "cadastral_rigid_located", 0.9, 5.0,
            m1="test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_KANDAMPALAYAM _9.dxf")

    def _stub_accept(r, action):
        r.recommendation = "ACCEPT_CADASTRAL"
        return True, "stub accepted"

    tools = {t.name: t for t in build_tools(
        [xv], {"village": "INGUR", "reattempt": _stub_accept})}
    out = tools["propose_fix"].fn("9", "road_closure_recover")
    assert out["accepted_by_gate"] is False and xv.recommendation == "REVIEW"


def test_geoagent_bridge_is_optional():
    # GeoAgent is not a hard dependency: the bridge degrades to None when absent.
    from landintel.agents.geoagent_adapter import build_geoagent, geoagent_available
    if not geoagent_available():
        assert build_geoagent([_r("724", "REVIEW")], {}) is None


def test_verification_flags_cross_village_confident():
    # A KANDAMPALAYAM plot wrongly marked confident must FAIL the FP invariant.
    results = [_r("9", "ACCEPT_CADASTRAL", "cadastral_rigid", 0.8, 5.0,
                  m1="test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_KANDAMPALAYAM _9.dxf")]
    rep = VerificationAgent().run(results, {"village": "INGUR"})
    assert rep.failed
    assert any(c.name == "fp_no_cross_village_in_confident"
               and c.severity == Severity.FAIL for c in rep.checks)


# ------------------------------------------- A2: demote mirror-write must be LOUD --
def test_demote_mirror_write_failure_raises(caplog):
    # If the raw result refuses the REVIEW mirror-write, the disposition and the
    # deliverable set silently diverge (verified-REVIEW vs shipped-ACCEPT). That
    # must raise, never pass.
    import logging

    import pytest

    from landintel.agents.dispositions import PlotDisposition

    class _FrozenRaw:
        recommendation = "ACCEPT"

        @property
        def note(self):
            return ""

        @note.setter
        def note(self, _):
            raise AttributeError("frozen")

    d = PlotDisposition(survey="X", recommendation="ACCEPT", raw=_FrozenRaw())
    with caplog.at_level(logging.WARNING, logger="landintel.agents.dispositions"):
        with pytest.raises(AttributeError):
            d.demote("test failure")
    assert any("mirror-write FAILED" in r.message for r in caplog.records)


def test_demote_never_promotes_and_mirrors_on_success():
    from landintel.agents.dispositions import PlotDisposition

    class _Raw:
        recommendation = "ACCEPT"
        note = ""

    raw = _Raw()
    d = PlotDisposition(survey="Y", recommendation="ACCEPT", raw=raw)
    d.demote("overlap")
    assert d.recommendation == "REVIEW" and raw.recommendation == "REVIEW"
    # demoting a non-confident plot is a no-op (never promotes, never flips states)
    d2 = PlotDisposition(survey="Z", recommendation="NO_COVERAGE")
    d2.demote("noop")
    assert d2.recommendation == "NO_COVERAGE"


# --------------------------------------- A3: single-source cross-village helper --
def test_is_cross_village_helper_is_shared():
    import inspect

    from landintel.agents import dispositions, input_request, llm_assist, verification

    assert hasattr(dispositions, "is_cross_village")
    for mod in (verification, input_request, llm_assist):
        # the local 9-line copies are gone; each module aliases the shared one
        assert mod._is_cross_village is dispositions.is_cross_village
        assert "def _is_cross_village" not in inspect.getsource(mod)


def test_is_cross_village_conservative_defaults():
    from landintel.agents.dispositions import PlotDisposition, is_cross_village

    d = PlotDisposition(survey="1", recommendation="REVIEW")   # no m1_file
    assert is_cross_village(d, "INGUR") is False               # conservative
    assert is_cross_village(d, None) is False
    d2 = PlotDisposition(
        survey="9", recommendation="REVIEW",
        m1_file="test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_KANDAMPALAYAM _9.dxf")
    assert is_cross_village(d2, "INGUR") is True               # real mismatch caught


# ------------------------------------------------- A4: AGENTS BEGIN summary log --
def test_run_agent_layer_logs_survey_summary(tmp_path, caplog):
    import logging

    from landintel.agents.orchestrator import run_agent_layer

    results = [_r("763", "ACCEPT_CADASTRAL", "cadastral_rigid", 1.0, 2.0),
               _r("668", "REVIEW", "cadastral_rigid_located", 0.78, 19.0)]
    with caplog.at_level(logging.INFO, logger="landintel.agents.orchestrator"):
        run_agent_layer(results, tmp_path,
                        {"village": "INGUR", "enable_auto_regate": False,
                         "enable_memory": False})
    assert any("AGENTS BEGIN" in r.message and "763" in r.message
               and "668" in r.message for r in caplog.records)


# --------------------------------------------- A7: LLM_ORDER strips every item --
def test_llm_order_env_all_items_stripped(monkeypatch):
    # The adapter must strip EVERY item so "local, claude , manus" parses cleanly.
    import inspect

    from landintel.agents import geoagent_adapter

    src = inspect.getsource(geoagent_adapter.build_geoagent)
    assert "p.strip() for p in" in src            # strip applied per-item
    assert "order[0].strip()" not in src          # not just the first
