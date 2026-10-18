# Text chat prototype

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent import config
from agent.brain import Brain
from agent.memory import CallLog


def main():
    try:
        brain = Brain()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return
    log = CallLog()
    print(f"Thathayya — text chat (type 'q' to quit, 'reset' to clear history, '/remember <fact>' to teach)")
    print(f"learned memories: {brain.memory.count()}  |  log: {log.path.name}")
    while True:
        try:
            q = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not q.strip():
            continue
        low = q.strip().lower()
        if low in {"q", "exit", "quit"}:
            break
        if low == "reset":
            brain.reset()
            print("(history cleared)")
            continue
        if low.startswith("/remember"):
            fact = q.strip()[len("/remember"):].strip()
            if not fact:
                print("usage: /remember <fact>")
                continue
            saved = brain.remember(fact)
            log.write("learn", fact)
            print(f"(remembered: {saved})")
            continue
        log.write("user", q)
        try:
            reply = brain.chat(q)
            log.write("assistant", reply)
            print(reply)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    print("bye")


if __name__ == "__main__":
    main()
