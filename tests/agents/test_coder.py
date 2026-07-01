"""Qwen local coding agent -- "Claude Code, but local Qwen".

The live model is mocked (scripted llm) so these pin the SAFETY-CRITICAL logic: tools dispatch
correctly, paths are confined to the repo root, the permission gate blocks unapproved actions,
readonly mode refuses actions, the ReAct loop runs to a final, and a tool error never crashes.
The agent operates the repo only; it cannot place a plot (no pipeline tools here).
"""
from __future__ import annotations

import json

from landintel.llm.coder import CodingAgent, CoderResult


def _scripted_llm(replies):
    """Return an llm(prompt, max_tokens, system) that yields the given replies in order."""
    it = iter(replies)

    def llm(prompt, max_tokens=0, system=None):
        try:
            return (next(it), "stub")
        except StopIteration:
            return ('{"final": "done"}', "stub")
    return llm


def _agent(tmp_path, replies, **kw):
    return CodingAgent(root=tmp_path, llm=_scripted_llm(replies), **kw)


# --- read tools auto-run; loop reaches a final ------------------------------
def test_read_file_then_finish(tmp_path):
    (tmp_path / "hello.txt").write_text("line1\nline2\n")
    a = _agent(tmp_path, [
        json.dumps({"tool": "read_file", "args": {"path": "hello.txt"}}),
        json.dumps({"final": "the file has 2 lines"}),
    ])
    res = a.run("read hello.txt")
    assert isinstance(res, CoderResult)
    assert res.final == "the file has 2 lines"
    assert res.steps[0].tool == "read_file"
    assert "line1" in res.steps[0].observation["content"]


def test_glob_and_grep(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("x = 2\n")
    a = _agent(tmp_path, [
        json.dumps({"tool": "glob", "args": {"pattern": "*.py"}}),
        json.dumps({"tool": "grep", "args": {"pattern": "def ", "glob": "*.py"}}),
        json.dumps({"final": "found it"}),
    ])
    res = a.run("find python defs")
    assert set(res.steps[0].observation["matches"]) == {"a.py", "b.py"}
    assert any("a.py" in m for m in res.steps[1].observation["matches"])


# --- path confinement -------------------------------------------------------
def test_path_outside_root_is_refused(tmp_path):
    a = _agent(tmp_path, [
        json.dumps({"tool": "read_file", "args": {"path": "../../etc/passwd"}}),
        json.dumps({"final": "blocked"}),
    ])
    res = a.run("read a system file")
    assert "error" in res.steps[0].observation
    assert "outside the repo root" in res.steps[0].observation["error"]


# --- permission gate --------------------------------------------------------
def test_run_requires_approval_and_can_be_declined(tmp_path):
    calls = {"n": 0}

    def deny(kind, detail):
        calls["n"] += 1
        return False

    a = _agent(tmp_path, [
        json.dumps({"tool": "run", "args": {"cmd": "echo hi"}}),
        json.dumps({"final": "was declined"}),
    ], mode="ask", approve=deny)
    res = a.run("run echo")
    assert calls["n"] == 1                       # approval was asked
    assert res.steps[0].approved is False
    assert "refused" in res.steps[0].observation


def test_write_file_approved_writes(tmp_path):
    a = _agent(tmp_path, [
        json.dumps({"tool": "write_file", "args": {"path": "out.txt", "content": "hello"}}),
        json.dumps({"final": "wrote it"}),
    ], mode="ask", approve=lambda k, d: True)
    res = a.run("create out.txt")
    assert res.steps[0].observation.get("written") is True
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_readonly_mode_refuses_actions_without_prompting(tmp_path):
    asked = {"n": 0}
    a = _agent(tmp_path, [
        json.dumps({"tool": "write_file", "args": {"path": "x.txt", "content": "no"}}),
        json.dumps({"final": "refused"}),
    ], mode="readonly", approve=lambda k, d: asked.__setitem__("n", asked["n"] + 1) or True)
    res = a.run("try to write")
    assert asked["n"] == 0                        # never even asked
    assert res.steps[0].approved is False
    assert not (tmp_path / "x.txt").exists()


def test_auto_mode_runs_without_asking(tmp_path):
    a = _agent(tmp_path, [
        json.dumps({"tool": "write_file", "args": {"path": "auto.txt", "content": "yes"}}),
        json.dumps({"final": "done"}),
    ], mode="auto", approve=lambda k, d: (_ for _ in ()).throw(AssertionError("should not ask")))
    a.run("write auto.txt")
    assert (tmp_path / "auto.txt").read_text() == "yes"


# --- edit_file exact-match guard --------------------------------------------
def test_edit_file_unique_match(tmp_path):
    (tmp_path / "c.py").write_text("a = 1\nb = 2\n")
    a = _agent(tmp_path, [
        json.dumps({"tool": "edit_file", "args": {"path": "c.py", "old": "a = 1", "new": "a = 9"}}),
        json.dumps({"final": "edited"}),
    ], mode="auto")
    a.run("change a")
    assert (tmp_path / "c.py").read_text() == "a = 9\nb = 2\n"


def test_edit_file_missing_old_reports_error(tmp_path):
    (tmp_path / "d.py").write_text("a = 1\n")
    a = _agent(tmp_path, [
        json.dumps({"tool": "edit_file", "args": {"path": "d.py", "old": "zzz", "new": "q"}}),
        json.dumps({"final": "couldn't"}),
    ], mode="auto")
    res = a.run("edit missing")
    assert "not found" in res.steps[0].observation["error"]


# --- offline + bad JSON safety ----------------------------------------------
def test_offline_is_graceful(tmp_path):
    a = CodingAgent(root=tmp_path, llm=lambda *a, **k: None)
    res = a.run("anything")
    assert res.provider == "none" and "offline" in res.final.lower()


def test_unknown_tool_does_not_crash(tmp_path):
    a = _agent(tmp_path, [
        json.dumps({"tool": "nuke", "args": {}}),
        json.dumps({"final": "no such tool"}),
    ])
    res = a.run("call a fake tool")
    assert "unknown tool" in res.steps[0].observation["error"]
