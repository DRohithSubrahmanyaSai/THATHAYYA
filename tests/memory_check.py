"""Prove long-term memorisation: facts persist across simulated calls.

Simulates call 1 (teach a fact) then a fresh call 2 and asserts the fact
is still recalled. Also confirms the dataset.json facts are present in the
system prompt at startup.

Run:
    .\.venv\Scripts\python.exe tests/memory_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import config
from agent.brain import Brain
from agent.memory import Memory

# A fact to teach during "call 1".
TEACH = "నా మిత్రు పేరూ కరణం, గురతే పేటేతూకొం"


def main() -> None:
    ok = True

    # Startup: dataset facts must be in the system prompt.
    if not config.DATASET_PROMPT:
        print("FAIL: dataset prompt is empty")
        sys.exit(1)
    for must in ("Kompatla Satyanarayana", "Jinnuru", "Varalakshmi"):
        if must not in config.DATASET_PROMPT:
            print(f"FAIL: dataset prompt missing '{must}'")
            ok = False

    # Call 1: teach a new fact (goes to pending, then auto-kept).
    m1 = Memory()
    m1.add_pending(TEACH)
    kept = m1.promote_all()
    print(f"call 1: noted + auto-kept {kept} fact(s)")

    # Call 2: FRESH Memory + Brain (simulating a new process/startup).
    m2 = Memory()
    recalled = TEACH in "\n".join(m2.facts()) or any(
        "కరణం" in f for f in m2.facts()
    )
    b2 = Brain()
    prompt_has = "కరణం" in b2.system or "Karana" in b2.system

    print(f"call 2 (fresh): fact recalled in learned memories = {recalled}")
    print(f"call 2 (fresh): fact in system prompt     = {prompt_has}")

    if not (recalled and prompt_has):
        ok = False

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
