import re
from pathlib import Path

from . import config

MAX_KEYTERMS = 100

_COMMON = {
    "the", "and", "his", "her", "our", "you", "your", "they", "them", "that",
    "this", "with", "from", "name", "village", "daughter", "son", "wife",
    "mother", "father", "grandson", "grandfather", "thathayya",
}


def _read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    text = path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def extract_terms(text: str) -> list[str]:
    terms = []
    for tok in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", text):
        if tok[0].isupper() and tok.lower() not in _COMMON:
            terms.append(tok)
    return terms


def get_keyterms() -> list[str]:
    terms = list(_read_list(config.KEYTERMS_FILE))
    if config.MEMORY_FILE.exists():
        fact_lines = [
            ln for ln in config.MEMORY_FILE.read_text(encoding="utf-8").splitlines() if ln.strip().startswith("- ")
        ]
        terms += extract_terms("\n".join(fact_lines))
    seen = set()
    unique = []
    for t in terms:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique[:MAX_KEYTERMS]
