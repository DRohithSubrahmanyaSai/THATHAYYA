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
    print("Thathayya - text chat (q=quit, reset=clear history, /remember <fact>=teach)")
    print("Pending: /pending list | /keep [fact|all] promote | /drop [fact|all] discard")
    print("Pronunciation: /pron list | /pron add <wrong> = <right> | /pron remove <wrong>")
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
        if low.startswith("/pron"):
            rest = q.strip()[len("/pron"):].strip()
            if not rest or rest.lower() == "list":
                items = brain.pron.items()
                if not items:
                    print("(no pronunciation rules)")
                else:
                    for it in items:
                        print(f"  {it['text']} -> {it['pronunciation']}")
                continue
            sub = rest.lower()
            if sub.startswith("remove "):
                word = rest[len("remove"):].strip()
                ok = brain.pron.remove(word)
                brain.refresh_system()
                print(f"(removed: {word})" if ok else f"(not found: {word})")
                continue
            if sub.startswith("add "):
                body = rest[len("add"):].strip()
                if "->" in body:
                    wrong, right = body.split("->", 1)
                elif "=" in body:
                    wrong, right = body.split("=", 1)
                else:
                    print("usage: /pron add <wrong> = <right>")
                    continue
                if brain.pron.add(wrong.strip(), right.strip()):
                    brain.refresh_system()
                    print(f"(pronunciation saved: {wrong.strip()} -> {right.strip()})")
                else:
                    print("(rejected: need 2+ chars on the wrong side and a real change)")
                continue
            print("usage: /pron list | /pron add <wrong> = <right> | /pron remove <wrong>")
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
