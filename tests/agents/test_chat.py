"""Qwen chat console -- terminal chat to the brain, for the operator AND Claude.

The live model is mocked (injected llm) so these pin the PURE logic: the system prompt carries
the concept + persisted knowledge, history accumulates, sender is tagged, every turn persists to
the memory graph, offline degrades gracefully, and slash-commands dispatch. Safety: chat never
places a plot (it only talks).
"""
from __future__ import annotations

from landintel.llm.chat import QwenChat, handle_command
from landintel.llm.memory_graph import MemoryGraph


def _graph(tmp_path):
    g = MemoryGraph(tmp_path / "mem.json")
    g.record_knowledge("hard_rule", "only math gates accept; the brain only proposes", ["core"])
    return g


def test_system_prompt_carries_concept_and_knowledge(tmp_path):
    chat = QwenChat(graph=_graph(tmp_path), llm=lambda *a, **k: ("ok", "stub"))
    assert "LandIntel" in chat.system
    assert "HARD RULE" in chat.system
    assert "hard_rule: only math gates accept" in chat.system   # seeded knowledge present
    assert "INTERACTIVE CONSOLE" in chat.system                  # chat preamble present


def test_send_accumulates_history_and_tags_sender(tmp_path):
    seen = {}

    def fake_llm(prompt, max_tokens=0, system=None):
        seen["prompt"] = prompt
        return ("hello back", "stub")

    chat = QwenChat(graph=_graph(tmp_path), llm=fake_llm)
    reply, provider = chat.send("hi from operator", sender="operator")
    assert reply == "hello back" and provider == "stub"
    # a second turn from Claude must see the prior turn in the rendered prompt
    chat.send("claude here", sender="claude")
    assert "Operator: hi from operator" in seen["prompt"]
    assert "Assistant: hello back" in seen["prompt"]
    assert "Claude: claude here" in seen["prompt"]
    assert len(chat.history) == 4                                 # 2 user + 2 assistant


def test_turns_persist_to_memory_graph(tmp_path):
    g = _graph(tmp_path)
    chat = QwenChat(graph=g, llm=lambda *a, **k: ("noted", "stub"))
    chat.send("remember this", sender="claude")
    turns = g.recall_chat()
    assert len(turns) == 1
    assert turns[0]["sender"] == "claude" and turns[0]["reply"] == "noted"
    # a fresh graph object on the same file still has it (cross-session)
    assert MemoryGraph(tmp_path / "mem.json").recall_chat()[0]["message"] == "remember this"


def test_offline_degrades_gracefully(tmp_path):
    chat = QwenChat(graph=_graph(tmp_path), llm=lambda *a, **k: None)   # all providers down
    reply, provider = chat.send("are you there?")
    assert provider == "none"
    assert "offline" in reply.lower()
    assert len(chat.history) == 2                                 # still recorded the turn


def test_slash_commands_dispatch(tmp_path):
    chat = QwenChat(graph=_graph(tmp_path), llm=lambda *a, **k: ("x", "stub"))
    assert handle_command("hello", chat) is None                 # not a command
    assert "commands:" in handle_command("/help", chat)
    assert "hard_rule" in handle_command("/knowledge", chat)
    assert handle_command("/quit", chat) == "__quit__"
    chat.send("a turn")
    assert "(conversation reset)" == handle_command("/reset", chat)
    assert chat.history == []                                     # reset cleared history
    assert "usage:" in handle_command("/recall", chat)           # missing arg guarded
