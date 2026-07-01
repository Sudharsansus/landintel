"""Interactive CHAT with the local Qwen brain -- terminal console for the operator AND Claude.

A small REPL (and single-shot CLI) that lets you talk to the LandIntel brain. Qwen answers with
the FULL concept + the persistent project knowledge already loaded as its system prompt, and every
turn is saved to the memory graph so the conversation survives across sessions. Either party can
drive it:
  * the human operator, interactively:   python qwen_chat.py
  * Claude (or any script), single-shot:  python qwen_chat.py --from claude "your question"

SAFETY: this is just conversation. The brain reads/explains/proposes/remembers; it CANNOT place a
plot or change a gate (the deterministic math gates still decide every placement). If Qwen/Ollama
is offline, the chat degrades to a clear message instead of crashing.

Slash-commands in the REPL: /help /status /knowledge /recall <survey> [village] /history /reset /quit
"""

from __future__ import annotations

import sys
import time
from typing import Callable

from .concept import SYSTEM_CONCEPT
from .memory_graph import MemoryGraph, default_graph
from .providers import llm_call, local_llm_status

_MAX_TURNS = 16          # how many prior turns to feed back as context
_OFFLINE = ("[no LLM provider reachable — Qwen/Ollama appears offline. Start it "
            "(`ollama serve` + `ollama pull qwen2.5:7b`) or run `python teach_qwen.py` "
            "to check status. The pipeline itself still runs deterministically without it.]")

_CHAT_PREAMBLE = (
    "\n\nYOU ARE IN AN INTERACTIVE CONSOLE. Messages may come from the human OPERATOR or from "
    "CLAUDE (the dev agent) — answer either, concisely and concretely. You are the LandIntel "
    "brain: explain, diagnose, and propose, but NEVER claim to place a plot or accept a "
    "placement (only the math gates do that). Ground every answer in the concept + project "
    "knowledge above; if you don't know, say so."
)


class QwenChat:
    """A stateful conversation with the brain, persisted to the memory graph."""

    def __init__(self, *, graph: MemoryGraph | None = None, extra_system: str = "",
                 max_turns: int = _MAX_TURNS, llm: Callable | None = None):
        self.graph = graph if graph is not None else default_graph()
        self.max_turns = max_turns
        self._llm = llm or llm_call
        self.system = self._build_system(extra_system)
        self.history: list[tuple[str, str]] = []      # (role, text), role in {operator,claude,assistant}
        self.session_id = f"chat:{int(time.time())}"

    def _build_system(self, extra: str) -> str:
        facts = self.graph.recall_knowledge()
        s = SYSTEM_CONCEPT
        if facts:
            s += "\n\nPROJECT KNOWLEDGE (carry this forward):\n" + "\n".join(
                f"- {f['topic']}: {f['fact']}" for f in facts)
        s += _CHAT_PREAMBLE
        if extra:
            s += "\n\n" + extra
        return s

    def _render_prompt(self, message: str, sender: str) -> str:
        lines = []
        for role, text in self.history[-self.max_turns * 2:]:
            tag = "Assistant" if role == "assistant" else role.capitalize()
            lines.append(f"{tag}: {text}")
        lines.append(f"{sender.capitalize()}: {message}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def send(self, message: str, sender: str = "operator", max_tokens: int = 500
             ) -> tuple[str, str]:
        """Send one message; return (reply, provider). ``sender`` is 'operator' or 'claude'."""
        self.history.append((sender, message))
        out = self._llm(self._render_prompt(message, sender), max_tokens=max_tokens,
                        system=self.system)
        reply, provider = out if out else (_OFFLINE, "none")
        self.history.append(("assistant", reply))
        try:
            self.graph.record_chat(sender, message, reply, self.session_id)
        except Exception:  # noqa: BLE001 - persistence must never break the chat
            pass
        return reply, provider

    def reset(self) -> None:
        self.history.clear()
        self.session_id = f"chat:{int(time.time())}"


def handle_command(line: str, chat: QwenChat) -> str | None:
    """Dispatch a slash-command. Returns the text to print, or None if ``line`` isn't a command."""
    if not line.startswith("/"):
        return None
    parts = line.split()
    cmd = parts[0].lower()
    if cmd in ("/help", "/h", "/?"):
        return ("commands: /status  /knowledge  /recall <survey> [village]  /history  "
                "/reset  /quit")
    if cmd == "/status":
        st = local_llm_status()
        return (f"Qwen reachable={st['reachable']} host={st['host']} "
                f"model={st.get('wanted_model')} present={st.get('model_present')}")
    if cmd == "/knowledge":
        facts = chat.graph.recall_knowledge()
        return "\n".join(f"- {f['topic']}: {f['fact']}" for f in facts) or "(no knowledge yet)"
    if cmd == "/recall":
        if len(parts) < 2:
            return "usage: /recall <survey_number> [village]"
        village = parts[2] if len(parts) > 2 else "INGUR"
        import json
        return json.dumps(chat.graph.recall(parts[1], village), indent=2)
    if cmd == "/history":
        if not chat.history:
            return "(no turns yet)"
        return "\n".join(f"{r}: {t[:160]}" for r, t in chat.history[-12:])
    if cmd == "/reset":
        chat.reset()
        return "(conversation reset)"
    if cmd in ("/quit", "/exit", "/q"):
        return "__quit__"
    return f"unknown command {cmd!r} — try /help"


def run_repl() -> None:
    chat = QwenChat()
    st = local_llm_status()
    print("=" * 64)
    print("LandIntel brain chat (Qwen). Type /help for commands, /quit to exit.")
    print(f"Qwen: reachable={st['reachable']} model={st.get('wanted_model')} "
          f"present={st.get('model_present')}")
    if not st["reachable"]:
        print("NOTE: Qwen offline — replies will say so until Ollama is up.")
    print("=" * 64)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(bye)")
            return
        if not line:
            continue
        out = handle_command(line, chat)
        if out is not None:
            if out == "__quit__":
                print("(bye)")
                return
            print(out)
            continue
        reply, provider = chat.send(line, sender="operator")
        print(f"qwen[{provider}]> {reply}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    sender = "operator"
    if argv and argv[0] == "--from":
        sender = argv[1] if len(argv) > 1 else "operator"
        argv = argv[2:]
    if argv:                                  # single-shot (Claude / scripted)
        chat = QwenChat()
        reply, _provider = chat.send(" ".join(argv), sender=sender)
        print(reply)                          # plain, pipe-friendly
    else:                                     # interactive REPL (operator)
        run_repl()


if __name__ == "__main__":
    main()
