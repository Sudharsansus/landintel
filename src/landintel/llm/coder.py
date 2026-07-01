"""Qwen as a local CODING/OPS AGENT -- "Claude Code, but local Qwen".

A dependency-free ReAct loop that lets the local Qwen READ, SEARCH, RUN, and EDIT inside the
repo to carry out a task -- the same shape as Claude Code, powered by the offline brain. Each
turn the model emits STRICT JSON: call a tool, or finish.

SAFETY MODEL (this is the important part -- a local 7B with shell + write access):
  * read-only tools (read_file / list_dir / glob / grep) auto-run.
  * action tools (run / write_file / edit_file) go through a PERMISSION GATE -- by default the
    operator must approve each one (y/N), exactly like Claude Code. Modes:
      - "ask"      (default): prompt before every action tool.
      - "readonly": action tools are refused (explore/analyse only -- safe to run unattended).
      - "auto"     : run actions without prompting (opt-in; prints a loud warning).
  * every file path is confined to the repo root; `run` is whatever the operator approves.
This agent OPERATES THE REPO; it does NOT touch pipeline placement decisions -- the math gates
still decide every plot (0-FP). Editing gate code is a normal dev action, gated by your approval.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .providers import llm_call

READ_TOOLS = {"read_file", "list_dir", "glob", "grep", "fetch_url"}
ACTION_TOOLS = {"run", "write_file", "edit_file"}
_ALL_TOOLS = READ_TOOLS | ACTION_TOOLS

_FETCH_TIMEOUT = 15       # seconds for a web fetch
_FETCH_MAX = 20000        # chars of fetched page kept (before the _MAX_OBS truncation)

_MAX_OBS = 4000           # chars of any tool observation fed back to the model
_RUN_TIMEOUT = 120        # seconds for a `run` shell command

_SYSTEM = """\
You are the LandIntel local coding agent -- a Claude-Code-style assistant powered by a LOCAL
model, operating inside this repository. You accomplish the operator's task by CALLING TOOLS, one
at a time, observing the result, and continuing until done.

Each turn reply with STRICT JSON and NOTHING else, one of:
  {"thought": "<short reasoning>", "tool": "<name>", "args": { ... }}
  {"final": "<answer / summary of what you did>"}

TOOLS:
  read_file   args: {"path": str, "offset": int=0, "limit": int=200}   # read a file (line range)
  list_dir    args: {"path": str="."}                                   # list a directory
  glob        args: {"pattern": str}                                    # e.g. "src/**/*.py"
  grep        args: {"pattern": str, "path": str=".", "glob": str?}     # search file contents
  fetch_url   args: {"url": str}            # fetch a web page -> text (read-only; needs network)
  run         args: {"cmd": str}            # shell command (needs operator approval)
  write_file  args: {"path": str, "content": str}    # overwrite/create (needs approval)
  edit_file   args: {"path": str, "old": str, "new": str}  # exact replace (needs approval)

RULES: paths stay inside the repo. Prefer read/search before acting. Make the SMALLEST change
that satisfies the task. When the task is done (or you have the answer), reply with {"final": ...}.
Do not invent file contents -- read first. Keep thoughts short.
"""


@dataclass
class Step:
    tool: str
    args: dict
    observation: dict
    approved: bool = True


@dataclass
class CoderResult:
    final: str
    steps: list[Step] = field(default_factory=list)
    provider: str = ""

    @property
    def tool_calls(self) -> int:
        return len(self.steps)


def _truncate(s: str, n: int = _MAX_OBS) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


class CodingAgent:
    """Drive Qwen through read/search/run/edit tools to do a task, with a permission gate."""

    def __init__(self, root: str | Path = ".", *, mode: str = "ask",
                 approve: Callable[[str, str], bool] | None = None,
                 llm: Callable | None = None, max_steps: int = 16):
        self.root = Path(root).resolve()
        self.mode = mode                       # "ask" | "readonly" | "auto"
        self._approve = approve or self._cli_approve
        self._llm = llm or llm_call
        self.max_steps = max_steps
        self.history: list[str] = []           # running convo for follow-ups

    # ----------------------------------------------------------------- safety ----
    def _safe(self, path: str) -> Path:
        p = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"path {path!r} is outside the repo root")
        return p

    @staticmethod
    def _cli_approve(kind: str, detail: str) -> bool:
        print(f"\n  >>> {kind} requested:\n{detail}")
        try:
            return input("  approve? [y/N] ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _gate(self, tool: str, detail: str) -> bool:
        if tool in READ_TOOLS:
            return True
        if self.mode == "readonly":
            return False
        if self.mode == "auto":
            return True
        return self._approve(tool, detail)

    # ----------------------------------------------------------------- tools -----
    def _exec(self, tool: str, args: dict) -> tuple[dict, bool]:
        try:
            if tool == "read_file":
                p = self._safe(args["path"])
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                off = int(args.get("offset", 0)); lim = int(args.get("limit", 200))
                body = "\n".join(lines[off:off + lim])
                return {"path": str(p), "lines": len(lines), "content": _truncate(body)}, True
            if tool == "list_dir":
                p = self._safe(args.get("path", "."))
                return {"path": str(p), "entries": sorted(
                    e.name + ("/" if e.is_dir() else "") for e in p.iterdir())[:300]}, True
            if tool == "glob":
                matches = [str(m.relative_to(self.root)) for m in
                           sorted(self.root.glob(args["pattern"]))[:200]]
                return {"pattern": args["pattern"], "matches": matches}, True
            if tool == "grep":
                return self._grep(args), True
            if tool == "fetch_url":
                return self._fetch_url(args["url"]), True

            # ---- action tools: permission gate ----
            if tool == "run":
                cmd = args["cmd"]
                if not self._gate("run", f"  $ {cmd}"):
                    return {"refused": "operator declined / read-only mode", "cmd": cmd}, False
                pr = subprocess.run(cmd, shell=True, cwd=self.root, capture_output=True,
                                    text=True, timeout=_RUN_TIMEOUT)
                return {"cmd": cmd, "returncode": pr.returncode,
                        "stdout": _truncate(pr.stdout), "stderr": _truncate(pr.stderr)}, True
            if tool == "write_file":
                p = self._safe(args["path"]); content = args.get("content", "")
                preview = content if len(content) < 600 else content[:600] + "\n...[more]"
                if not self._gate("write_file", f"  write {p}\n  ----\n{preview}\n  ----"):
                    return {"refused": "operator declined / read-only mode", "path": str(p)}, False
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return {"path": str(p), "bytes": len(content), "written": True}, True
            if tool == "edit_file":
                p = self._safe(args["path"]); old = args["old"]; new = args["new"]
                cur = p.read_text(encoding="utf-8")
                if old not in cur:
                    return {"error": "old text not found (must match exactly)", "path": str(p)}, True
                if cur.count(old) > 1:
                    return {"error": f"old text matches {cur.count(old)}x; make it unique"}, True
                if not self._gate("edit_file",
                                  f"  edit {p}\n  - {old[:200]}\n  + {new[:200]}"):
                    return {"refused": "operator declined / read-only mode", "path": str(p)}, False
                p.write_text(cur.replace(old, new, 1), encoding="utf-8")
                return {"path": str(p), "replaced": True}, True
            return {"error": f"unknown tool {tool!r}; valid: {sorted(_ALL_TOOLS)}"}, True
        except Exception as exc:  # noqa: BLE001 - a tool error never crashes the loop
            return {"error": str(exc)}, True

    def _grep(self, args: dict) -> dict:
        import re as _re
        pat = _re.compile(args["pattern"])
        base = self._safe(args.get("path", "."))
        g = args.get("glob") or "**/*"
        hits = []
        roots = [base] if base.is_file() else sorted(base.glob(g))
        for f in roots:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if pat.search(line):
                        hits.append(f"{f.relative_to(self.root)}:{i}: {line.strip()[:160]}")
                        if len(hits) >= 80:
                            return {"pattern": args["pattern"], "matches": hits, "capped": True}
            except Exception:  # noqa: BLE001
                continue
        return {"pattern": args["pattern"], "matches": hits}

    def _fetch_url(self, url: str) -> dict:
        """Fetch a web page and return readable text (read-only; stdlib only, no new deps).

        The local brain is offline-first, so this is the ONE tool that reaches the network --
        used to pull docs/reference pages on demand. HTML is reduced to text (scripts/styles
        stripped, tags removed); non-HTML is returned as-is. Any failure (no network, bad URL)
        returns an error dict rather than raising, so the agent loop keeps going.
        """
        import re as _re
        import urllib.request
        if not _re.match(r"^https?://", url):
            return {"error": "url must start with http:// or https://", "url": url}
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LandIntel-local-agent"})
            with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
                ctype = resp.headers.get_content_type()
                raw = resp.read(4 * 1024 * 1024)  # 4 MB cap
                charset = resp.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
            if "html" in ctype:
                text = _re.sub(r"(?is)<(script|style|head).*?</\1>", " ", text)
                text = _re.sub(r"(?s)<[^>]+>", " ", text)
                text = _re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&#39;|&quot;",
                               lambda m: {"&nbsp;": " ", "&amp;": "&", "&lt;": "<",
                                          "&gt;": ">", "&#39;": "'", "&quot;": '"'}[m.group()], text)
                text = _re.sub(r"[ \t]+", " ", text)
                text = _re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
            return {"url": url, "content_type": ctype, "content": _truncate(text[:_FETCH_MAX])}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"fetch failed: {exc}", "url": url}

    # ----------------------------------------------------------------- loop ------
    @staticmethod
    def _parse(text: str) -> dict | None:
        try:
            return json.loads(text[text.index("{"): text.rindex("}") + 1])
        except Exception:  # noqa: BLE001
            return None

    def run(self, task: str) -> CoderResult:
        """Run the ReAct loop for one task; returns the final answer + the steps taken."""
        convo = "\n".join(self.history[-8:])
        prompt = (f"{convo}\nTASK: {task}\nroot: {self.root}\nYour first JSON:").strip()
        out = self._llm(prompt, max_tokens=500, system=_SYSTEM)
        if out is None:
            return CoderResult(final="[Qwen offline — start Ollama. The agent needs the local "
                                     "model to plan tool calls.]", provider="none")
        reply, provider = out
        steps: list[Step] = []
        for _ in range(self.max_steps):
            step = self._parse(reply)
            if step is None:
                reply_out = self._llm(prompt + "\n\nREPLY WAS NOT VALID JSON. Resend ONE valid "
                                      "JSON object only.", max_tokens=500, system=_SYSTEM)
                if reply_out is None:
                    break
                reply = reply_out[0]
                continue
            if "final" in step:
                final = str(step["final"])
                self.history.append(f"TASK: {task}\nRESULT: {final[:400]}")
                return CoderResult(final=final, steps=steps, provider=provider)
            tool = step.get("tool", "")
            args = step.get("args") or {}
            obs, approved = self._exec(tool, args)
            steps.append(Step(tool=tool, args=args, observation=obs, approved=approved))
            convo2 = (f"TASK: {task}\n" + "".join(
                f"CALL: {json.dumps({'tool': s.tool, 'args': s.args})}\n"
                f"OBSERVATION: {_truncate(json.dumps(s.observation), 1200)}\n" for s in steps))
            nxt = self._llm(convo2 + "\nYour next JSON:", max_tokens=500, system=_SYSTEM)
            if nxt is None:
                break
            reply = nxt[0]
        return CoderResult(final="[reached step limit or model stopped emitting JSON; see steps]",
                           steps=steps, provider=provider)


# --------------------------------------------------------------------- CLI -------

def _banner(agent: CodingAgent, status: dict) -> None:
    print("=" * 66)
    print("LandIntel coding agent (local Qwen) — Claude-Code-style, offline.")
    print(f"root={agent.root}  mode={agent.mode}  "
          f"Qwen reachable={status['reachable']} model={status.get('wanted_model')}")
    if agent.mode == "auto":
        print("!! AUTO mode: action tools (run/write/edit) execute WITHOUT asking.")
    elif agent.mode == "readonly":
        print("readonly mode: explore/analyse only; run/write/edit are refused.")
    else:
        print("ask mode: you approve every run/write/edit (y/N).")
    print("Type a task; /quit to exit.")
    print("=" * 66)


def _print_result(res: CoderResult) -> None:
    for s in res.steps:
        tag = "" if s.approved else " [refused]"
        print(f"  - {s.tool}({json.dumps(s.args)[:80]}){tag}")
    print(f"qwen[{res.provider}]> {res.final}")


def main(argv: list[str] | None = None) -> None:
    import sys
    from .providers import local_llm_status
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = "ask"
    root = "."
    while argv and argv[0].startswith("--"):
        flag = argv.pop(0)
        if flag == "--readonly":
            mode = "readonly"
        elif flag in ("--auto", "--yolo"):
            mode = "auto"
        elif flag == "--root":
            root = argv.pop(0)
    agent = CodingAgent(root=root, mode=mode)
    status = local_llm_status()
    if argv:                                    # single-shot task
        _print_result(agent.run(" ".join(argv)))
        return
    _banner(agent, status)
    while True:
        try:
            task = input("\ntask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(bye)"); return
        if not task:
            continue
        if task in ("/quit", "/exit", "/q"):
            print("(bye)"); return
        _print_result(agent.run(task))


if __name__ == "__main__":
    main()
