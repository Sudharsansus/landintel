"""Chat with the local Qwen brain from the terminal.

Interactive (operator):   python qwen_chat.py
Single-shot (Claude/script): python qwen_chat.py --from claude "your question"

Thin entry point; all logic in landintel.llm.chat. The brain answers with the full LandIntel
concept + persistent project knowledge loaded, and saves every turn to the memory graph.
"""
from landintel.llm.chat import main

if __name__ == "__main__":
    main()
