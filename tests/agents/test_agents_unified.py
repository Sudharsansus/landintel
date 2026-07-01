"""Unified agent-layer tests -- every runtime agent on BOTH pipeline stages.

The agent layer must verify / build worklists / guard / reason for the new M2
(``list[ClubResult]``, FMB-only, in-memory footprints) AND the surveyor-matching M3
(``list[GeorefResult]``, footprints read from the output DXF) with the SAME code, via the
normalized ``PlotDisposition`` adapter. These tests assert that, plus the 0-FP invariants:
a planted overlapping pair is DEMOTED (never promoted), the worklist names the right
missing input, and no agent ever turns a non-ACCEPT into an ACCEPT.
"""
from __future__ import annotations

import numpy as np

from landintel.agents.dispositions import (
    PlotDisposition, from_club_result, from_georef_result, normalize)
from landintel.agents.guard import GuardAgent
from landintel.agents.input_request import InputRequestAgent
from landintel.agents.llm_assist import LLMAssistAgent
from landintel.agents.verification import VerificationAgent
from landintel.agents.base import InputType, Severity
from landintel.pipeline.m2_club.placement import CandidatePlacement, ClubResult
from landintel.pipeline.m2_georef.pipeline import GeorefResult


# --------------------------------------------------------------- builders ------
def _square(cx, cy, half=10.0):
    """A small square corner ring centred at (cx, cy), as an (N,2) numpy array."""
    return np.array([
        [cx - half, cy - half], [cx + half, cy - half],
        [cx + half, cy + half], [cx - half, cy + half],
    ], dtype=float)


def _placement(cx, cy, half=10.0, scale=1.0, area_ratio=1.0, passes=True, seed_ok=True):
    ring = _square(cx, cy, half)
    return CandidatePlacement(
        method="cadastral", R=np.eye(2), s=1.0, t=np.zeros(2),
        adjusted=ring, corner_ring=[0, 1, 2, 3], passes_gate=passes,
        area_ratio=area_ratio, rot_residual=2.0, scale=scale, seed_ok=seed_ok)


def _club(sn, rec, method="cadastral", cx=None, cy=None, conf=0.5, out="x.dxf",
          m1=None, note="", **pl):
    p = None
    if cx is not None:
        p = _placement(cx, cy, **pl)
    return ClubResult(
        m1_file=m1 or f"test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_INGUR_{sn}.dxf",
        survey_number=sn, recommendation=rec, method=method,
        output_file=out if p is not None else "", placement=p, confidence=conf, note=note)


def _georef(sn, rec, method="", cov=0.0, resid=float("inf"), m1=None):
    return GeorefResult(
        m1_file=m1 or f"test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_INGUR_{sn}.dxf",
        survey_number=sn, matched=True, recommendation=rec,
        match_method=method, chain_coverage=cov, cad_residual=resid)


# ============================================================ adapter ==========
def test_normalize_handles_both_result_types():
    mixed = [_club("724", "ACCEPT", cx=100.0, cy=100.0),
             _georef("763", "ACCEPT_CADASTRAL", "cadastral_rigid", 1.0, 2.0)]
    disps = normalize(mixed)
    assert all(isinstance(d, PlotDisposition) for d in disps)
    assert {d.source for d in disps} == {"club", "georef"}
    # a list already of PlotDisposition is returned unchanged
    assert normalize(disps) is not disps and len(normalize(disps)) == 2


def test_from_club_result_resolves_in_memory_footprint():
    d = from_club_result(_club("724", "ACCEPT", cx=0.0, cy=0.0, half=5.0))
    assert d.is_confident and d.has_geometry
    assert abs(d.footprint.area - 100.0) < 1e-6      # 10x10 square


def test_from_georef_split_of_chain_coverage_by_method():
    cad = from_georef_result(_georef("1", "REVIEW", "cadastral_rigid_located", 0.78, 19.0))
    geo = from_georef_result(_georef("2", "REVIEW", "geometric_6/7", 0.25))
    assert cad.area_ratio == 0.78 and cad.chain_coverage != cad.chain_coverage  # NaN
    assert geo.chain_coverage == 0.25 and geo.area_ratio != geo.area_ratio       # NaN


# ===================================================== VerificationAgent =======
def test_verification_demotes_overlapping_accept_pair_club():
    # Two ACCEPT clubs on (almost) the same footprint -> the lower-confidence one is
    # DEMOTED to REVIEW (never promoted) and the tiling invariant FAILS.
    a = _club("724", "ACCEPT", cx=0.0, cy=0.0, conf=0.9)
    b = _club("725", "ACCEPT", cx=2.0, cy=0.0, conf=0.4)   # overlaps a heavily
    rep = VerificationAgent().run([a, b], {"village": "INGUR"})
    assert rep.failed
    tiling = next(c for c in rep.checks if c.name == "fp_non_overlapping_tiling")
    assert tiling.severity == Severity.FAIL
    # demote-only: the LOWER-confidence raw ClubResult flipped to REVIEW; the other stays.
    assert b.recommendation == "REVIEW" and a.recommendation == "ACCEPT"


def test_verification_demotes_overlapping_accept_pair_georef(monkeypatch):
    # Same invariant on M3 results: patch the DXF footprint reader so two ACCEPTs overlap.
    import landintel.agents.dispositions as D
    from shapely.geometry import Polygon
    polys = {"a.dxf": Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
             "b.dxf": Polygon([(1, 0), (11, 0), (11, 10), (1, 10)])}
    monkeypatch.setattr(D, "_georef_footprint", lambda f: polys.get(f))
    a = _georef("724", "ACCEPT", "geometric_6/7", 0.9)
    a.output_file, a.confidence = "a.dxf", 0.9
    a.verify_result = type("V", (), {"all_passed": True})()
    b = _georef("725", "ACCEPT", "geometric_6/7", 0.6)
    b.output_file, b.confidence = "b.dxf", 0.6
    b.verify_result = type("V", (), {"all_passed": True})()
    rep = VerificationAgent().run([a, b], {"village": "INGUR"})
    assert rep.failed
    assert b.recommendation == "REVIEW" and a.recommendation == "ACCEPT"   # demote-only


def test_verification_accept_without_output_file_fails_club():
    # When the run writes deliverables, an ACCEPT missing its output file is not a
    # deliverable -> FAIL. (good has a footprint+output; bad is ACCEPT with neither.)
    good = _club("724", "ACCEPT", cx=0.0, cy=0.0)
    bad = _club("725", "ACCEPT")                   # cx=None -> no placement, no output_file
    rep = VerificationAgent().run([good, bad], {})
    c = next(c for c in rep.checks if c.name == "accept_has_output_file")
    assert c.severity == Severity.FAIL and rep.failed


def test_verification_passes_clean_tiling_club():
    # Two ACCEPTs on disjoint footprints + a REVIEW -> all invariants pass, nothing lost.
    a = _club("724", "ACCEPT", cx=0.0, cy=0.0)
    b = _club("725", "ACCEPT", cx=1000.0, cy=1000.0)
    c = _club("726", "REVIEW", cx=2000.0, cy=2000.0, note="below gate")
    rep = VerificationAgent().run([a, b, c], {"village": "INGUR"})
    assert not rep.failed
    cov = next(ch for ch in rep.checks if ch.name == "coverage_accounting")
    assert cov.severity == Severity.OK
    assert a.recommendation == "ACCEPT" and b.recommendation == "ACCEPT"


# ===================================================== InputRequestAgent =======
def test_input_request_worklist_club_names_right_input():
    results = [
        _club("724", "ACCEPT", cx=0.0, cy=0.0),                        # confident -> no req
        _club("668", "REVIEW", method="cadastral", cx=10.0, cy=10.0,   # borderline located
              area_ratio=1.0, passes=False, note="below gate"),
        _club("900", "NO_COVERAGE", method=""),                        # no position -> 2 corners
        _club("9", "NO_COVERAGE", method="",
              m1="test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_KANDAMPALAYAM _9.dxf"),  # x-village
    ]
    rep = InputRequestAgent().run(results, {"village": "INGUR"})
    by = {q.survey_number: q.input_type for q in rep.requests}
    assert "724" not in by                                  # confident -> no request
    assert by["668"] == InputType.CONFIRM_PLACEMENT         # located, just under the bar
    assert by["900"] == InputType.TWO_CORNER_SEED           # no position
    assert by["9"] == InputType.VILLAGE_REFERENCE           # different village
    # ORDERED most-impactful first: CONFIRM(0) < TWO_CORNER(2) < VILLAGE(3)
    order = [q.input_type for q in rep.requests]
    assert order == sorted(order, key=lambda t: {InputType.CONFIRM_PLACEMENT: 0,
                                                 InputType.TWO_CORNER_SEED: 2,
                                                 InputType.VILLAGE_REFERENCE: 3}[t])


def test_input_request_worklist_georef_unchanged():
    # The M3 mapping the original suite locks in must still hold through normalization.
    results = [
        _georef("763", "ACCEPT_CADASTRAL", "cadastral_rigid", 1.0, 2.0),
        _georef("668", "REVIEW", "cadastral_rigid_located", 0.78, 19.0),  # bad parcel
        _georef("699", "REVIEW", "geometric_6/7_inliers", 0.25),          # partial trace
        _georef("9", "NO_COVERAGE", "",
                m1="test2/INGUR/m1_outputs/FMB_ERODE_PERUNDURAI_KANDAMPALAYAM _9.dxf"),
    ]
    rep = InputRequestAgent().run(results, {"village": "INGUR"})
    by = {q.survey_number: q.input_type for q in rep.requests}
    assert "763" not in by
    assert by["668"] == InputType.CLEARER_PARCEL
    assert by["699"] == InputType.CONFIRM_PLACEMENT
    assert by["9"] == InputType.VILLAGE_REFERENCE
    assert len(rep.requests) == 3


# ============================================================== GuardAgent =====
def test_guard_flags_duplicate_confident_survey_club():
    a = _club("724", "ACCEPT", cx=0.0, cy=0.0)
    dup = _club("724", "ACCEPT", cx=5000.0, cy=5000.0)     # same survey, second ACCEPT
    rep = GuardAgent().run([a, dup], {})
    c = next(ch for ch in rep.checks if ch.name == "no_duplicate_confident_survey")
    assert c.severity == Severity.FAIL


def test_guard_flags_anagram_collision_club():
    # 669 and its 180-rotation 699 placed confidently <20 m apart -> hard FP.
    a = _club("669", "ACCEPT", cx=0.0, cy=0.0)
    b = _club("699", "ACCEPT", cx=5.0, cy=0.0)
    rep = GuardAgent().run([a, b], {})
    c = next(ch for ch in rep.checks if ch.name == "anagram_no_confident_collision")
    assert c.severity == Severity.FAIL


def test_guard_two_confident_on_same_footprint_club():
    a = _club("724", "ACCEPT", cx=0.0, cy=0.0)
    b = _club("725", "ACCEPT", cx=0.0, cy=0.0)             # identical footprint
    rep = GuardAgent().run([a, b], {})
    c = next(ch for ch in rep.checks if ch.name == "no_two_confident_on_same_footprint")
    assert c.severity == Severity.FAIL


def test_guard_clean_club_passes():
    a = _club("724", "ACCEPT", cx=0.0, cy=0.0)
    b = _club("888", "ACCEPT", cx=9000.0, cy=9000.0)
    rep = GuardAgent().run([a, b], {})
    assert not rep.failed


def test_guard_clean_georef_passes():
    # No output files -> no footprints -> geometry checks INFO/OK, anagram OK.
    a = _georef("763", "ACCEPT_CADASTRAL", "cadastral_rigid", 1.0, 2.0)
    rep = GuardAgent().run([a], {})
    assert not rep.failed


# ============================================================ LLMAssistAgent ===
def test_llm_assist_never_places_and_only_diagnoses_unconfident_club(monkeypatch):
    monkeypatch.setenv("LANDINTEL_LLM_ORDER", "")          # offline -> rule-based, fast
    results = [
        _club("724", "ACCEPT", cx=0.0, cy=0.0),                    # confident -> not reasoned
        _club("668", "REVIEW", method="cadastral", cx=10.0, cy=10.0,
              area_ratio=0.30, passes=False),
        _club("900", "NO_COVERAGE", method=""),
    ]
    before = [r.recommendation for r in results]
    rep = LLMAssistAgent().run(results, {"village": "INGUR"})
    assert [r.recommendation for r in results] == before           # agent never placed
    from landintel.agents.concept import SAFE_ACTIONS
    assert {p.survey_number for p in rep.proposals} == {"668", "900"}   # only unconfident
    assert all(p.action in SAFE_ACTIONS for p in rep.proposals)


def test_llm_assist_mixed_stage_list(monkeypatch):
    # A heterogeneous list (one ClubResult + one GeorefResult) is reasoned uniformly.
    monkeypatch.setenv("LANDINTEL_LLM_ORDER", "")
    mixed = [_club("668", "REVIEW", method="cadastral", cx=10.0, cy=10.0,
                   area_ratio=0.30, passes=False),
             _georef("699", "REVIEW", "geometric_6/7_inliers", 0.25)]
    rep = LLMAssistAgent().run(mixed, {"village": "INGUR"})
    assert {p.survey_number for p in rep.proposals} == {"668", "699"}
    from landintel.agents.concept import SAFE_ACTIONS
    assert all(p.action in SAFE_ACTIONS for p in rep.proposals)


# ===================================== cross-cutting: no agent ever promotes ===
def test_no_agent_promotes_a_non_accept(monkeypatch):
    monkeypatch.setenv("LANDINTEL_LLM_ORDER", "")
    non_accept = {"REVIEW", "NO_COVERAGE"}
    for build in (
        lambda: [_club("668", "REVIEW", method="cadastral", cx=10.0, cy=10.0,
                       area_ratio=0.30, passes=False),
                 _club("900", "NO_COVERAGE")],
        lambda: [_georef("668", "REVIEW", "cadastral_rigid_located", 0.78, 19.0),
                 _georef("900", "NO_COVERAGE", "")],
    ):
        for Agent in (VerificationAgent, GuardAgent, InputRequestAgent, LLMAssistAgent):
            results = build()
            before = {id(r): r.recommendation for r in results}
            Agent().run(results, {"village": "INGUR"})
            for r in results:
                if before[id(r)] in non_accept:
                    assert r.recommendation in non_accept   # never upgraded to ACCEPT
