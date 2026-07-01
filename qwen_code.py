"""Qwen as a local coding/ops agent -- "Claude Code, but local Qwen".

Interactive (operator):   python qwen_code.py            # ask-before-acting (default, safe)
Read-only explore:        python qwen_code.py --readonly
Autonomous (opt-in):      python qwen_code.py --auto      # runs actions WITHOUT asking
Single-shot task:         python qwen_code.py "summarise src/landintel/llm and list its modules"

The agent READS/SEARCHES/RUNS/EDITS inside the repo via a ReAct loop on the local Qwen. Action
tools (run/write/edit) are permission-gated; read tools auto-run. Thin entry; logic in
landintel.llm.coder.
"""
from landintel.llm.coder import main

if __name__ == "__main__":
    main()
