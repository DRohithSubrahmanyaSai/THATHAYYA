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
    print("Pending facts: '/pending' list · '/keep [fact|all]' promote · '/drop [fact|all]' discard")
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
        if low == "/pending":
            facts = brain.memory.pending()
            if not facts:
                print("(no pending facts)")
            else:
                for f in facts:
                    print(f"  - {f}")
            continue
        if low.startswith("/keep"):
            arg = q.strip()[len("/keep"):].strip().lower()
            if arg in ("", "all"):
                n = brain.memory.promote_all()
                print(f"(kept {n} fact(s) into learned memories)")
            else:
                ok = brain.memory.promote(arg)
                print(f"(kept: {arg})" if ok else f"(not found in pending: {arg})")
            continue
        if low.startswith("/drop"):
            arg = q.strip()[len("/drop"):].strip().lower()
            if arg in ("", "all"):
                brain.memory.discard_all()
                print("(dropped all pending)")
            else:
                ok = brain.memory.discard(arg)
                print(f"(dropped: {arg})" if ok else f"(not found in pending: {arg})")
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
