"""Fast checks for pronunciation learning + safe output rewriting (no network)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import config
from agent.pronunciation import PronunciationStore, extract_pair_regex

tmp = Path(tempfile.mkdtemp(prefix="thathayya_pron_"))
config.PRON_DICT_ENABLED = False
config.KEYTERMS_LEARNED_FILE = tmp / "keyterms_learned.txt"
config.PRON_REFS_DIR = tmp / "refs"


def check(name: str, cond: bool, detail: str = "") -> int:
    print(f"[{'OK ' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    return 0 if cond else 1


def main() -> int:
    fails = 0
    store = PronunciationStore(path=tmp / "pron.json")

    pair = extract_pair_regex("డాగ్ కాదు దగ్గు")
    fails += check("regex extract", pair == ("డాగ్", "దగ్గు"), repr(pair))

    pair2 = extract_pair_regex("చా కాదు చానా")
    fails += check("regex extract short", pair2 == ("చా", "చానా"), repr(pair2))

    fails += check("reject single-char wrong", store.add("క", "గ") is False)
    fails += check("reject protected word", store.add("అవును", "ఔను") is False)

    fails += check("add pair", store.add("డాగ్", "దగ్గు") is True)
    rewritten = store.apply("డాగ్ గారు వచ్చారు")
    fails += check("apply whole word", rewritten == "దగ్గు గారు వచ్చారు", rewritten)

    unchanged = store.apply("డాగ్గు అని రాశారు")
    fails += check("apply respects boundaries", unchanged == "డాగ్గు అని రాశారు", unchanged)

    fails += check("prompt rule present", 'డాగ్' in store.as_prompt(), store.as_prompt())

    ref = store.save_ref(b"\x00\x00" * 8000, 16000, "డాగ్", "దగ్గు")
    fails += check("save wav ref", bool(ref) and Path(ref).exists(), str(ref))

    reloaded = PronunciationStore(path=tmp / "pron.json")
    fails += check("persists to disk", reloaded.count() == 1, str(reloaded.items()))
    fails += check("keyterm appended", (tmp / "keyterms_learned.txt").exists())

    print(f"\n{'PASS' if not fails else 'FAIL'} ({fails} failure(s))")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
