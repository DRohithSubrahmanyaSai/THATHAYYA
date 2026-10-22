"""Fast checks for the live-memory layer (no network)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import config
from agent.memory import Memory, find_trigger

tmp = Path(tempfile.mkdtemp(prefix="thathayya_mem_"))
config.RUNTIME_DIR = tmp
config.LOGS_DIR = tmp / "logs"
config.LIVE_MEMORY_FILE = tmp / "live_memories.md"


def check(name: str, cond: bool, detail: str = "") -> int:
    print(f"[{'OK ' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    return 0 if cond else 1


def main() -> int:
    fails = 0
    m = Memory(path=tmp / "learned_memories.md", client=None)
    m.reset_live()

    m.add_live("మా మనవడు పేరు రోహిత్")
    fails += check("live fact visible immediately", "రోహిత్" in m.as_prompt(), m.as_prompt())
    fails += check("live not in long-term yet", "రోహిత్" not in "\n".join(m.facts()))

    m.add_pending("మా ఊరు జిన్నూరు")
    fails += check("pending not in long-term", "జిన్నూరు" not in "\n".join(m.facts()))
    fails += check("pending recorded", any("జిన్నూరు" in f for f in m.pending()))

    m.add_live("మా మనవడు పేరు రోహిత్")
    fails += check("live dedupe", len(m.live_facts()) == 1, str(m.live_facts()))

    kept = m.promote_all()
    fails += check("promote_all count", kept == 2, str(kept))
    learned = "\n".join(m.facts())
    fails += check("promoted to learned", "రోహిత్" in learned and "జిన్నూరు" in learned, learned)
    fails += check("live cleared", m.live_facts() == [])
    fails += check("pending cleared", m.pending() == [])

    fails += check("trigger telugu", find_trigger("గుర్తు పెట్టుకో, నా పేరు రోహిత్") is not None)
    fails += check("trigger none", find_trigger("ఈరోజు బాగుంది") is None)

    print(f"\n{'PASS' if not fails else 'FAIL'} ({fails} failure(s))")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
