# Text chat prototype

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent import config
from agent.brain import Brain


def main():
    try:
        brain = Brain()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return
    print("Thathayya — text chat (type 'q' to quit, 'reset' to clear history)")
    while True:
        try:
            q = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not q.strip():
            continue
        if q.strip().lower() in {"q", "exit", "quit"}:
            break
        if q.strip().lower() == "reset":
            brain.reset()
            print("(history cleared)")
            continue
        try:
            print(brain.chat(q))
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    print("bye")


if __name__ == "__main__":
    main()
