import json
import time
from pathlib import Path

from . import config

HEADER = (
    "# Learned memories — Thathayya\n\n"
    "> Facts taught to Thathayya during conversations. Auto-managed; safe to edit by hand.\n\n"
)

PENDING_HEADER = (
    "# Pending (noted, unverified) — Thathayya\n\n"
    "> Facts noted during the live call. You review; /keep promotes, /drop discards.\n\n"
)

LIVE_HEADER = (
    "# Live memories (this call) — Thathayya\n\n"
    "> Facts learned during the current call. Cleared at the start of each call.\n\n"
)


def pending_path() -> Path:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return config.RUNTIME_DIR / "pending_memories.md"


def live_path() -> Path:
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return config.LIVE_MEMORY_FILE


def _read_bullets(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("- "):
                out.append(line[2:].strip())
    except OSError:
        return []
    return out


class Memory:
    def __init__(self, path: Path | None = None, client=None, model: str | None = None):
        self.path = Path(path) if path else config.MEMORY_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(HEADER, encoding="utf-8")
        self.live_path = live_path()
        if not self.live_path.exists():
            self.live_path.write_text(LIVE_HEADER, encoding="utf-8")
        self.model = model or config.GROQ_MERGE_MODEL
        self._client = client
        if self._client is None and config.GROQ_API_KEY:
            from groq import Groq

            self._client = Groq(api_key=config.GROQ_API_KEY)

    # --- learned (long-term) ---
    def facts(self) -> list[str]:
        return _read_bullets(self.path)

    def count(self) -> int:
        return len(self.facts())

    # --- live (this call) ---
    def live_facts(self) -> list[str]:
        return _read_bullets(self.live_path)

    def reset_live(self) -> None:
        self.live_path.write_text(LIVE_HEADER, encoding="utf-8")

    def add_live(self, statement: str) -> str:
        statement = statement.strip()
        if not statement:
            return ""
        existing = self.live_facts()
        if statement in existing or statement in self.facts():
            return statement
        existing.append(statement)
        self._write_live(existing)
        return statement

    def _write_live(self, facts: list[str]) -> None:
        body = "\n".join(f"- {f}" for f in facts)
        self.live_path.write_text(LIVE_HEADER + body + "\n", encoding="utf-8")

    def as_prompt(self) -> str:
        seen = set()
        merged: list[str] = []
        for fact in self.facts() + self.live_facts():
            if fact and fact not in seen:
                seen.add(fact)
                merged.append(fact)
        if not merged:
            return ""
        return "\n".join(f"- {f}" for f in merged)

    # --- add / merge ---
    def add(self, statement: str, merge: bool = True) -> str:
        statement = statement.strip()
        if not statement:
            return ""
        existing = self.facts()
        if statement in existing:
            return statement
        if merge and self._client is not None and existing:
            try:
                merged = self._merge(existing, statement)
                if merged:
                    self._write(merged)
                    return merged[-1]
            except Exception:
                pass
        existing.append(statement)
        self._write(existing)
        return statement

    def _merge(self, existing: list[str], statement: str) -> list[str]:
        current = "\n".join(f"- {f}" for f in existing)
        prompt = (
            "You keep a short bullet list of durable facts about a man named Thathayya, "
            "as told by his grandson.\n\n"
            f"Existing facts:\n{current}\n\n"
            f'New statement from the grandson: "{statement}"\n\n'
            "Return the FULL updated bullet list. Rules:\n"
            "- One fact per line, each line starting with '- '.\n"
            "- Merge exact duplicates; if the new statement corrects an old fact, replace it.\n"
            "- KEEP any new name, place, job, preference, or person the grandson states — "
            "these are part of Thathayya's real story. Only drop a line if it is pure small talk "
            "that states no durable fact.\n"
            "- Keep each fact short. Keep the same language mix already used.\n"
            "- If the new statement has no durable fact, return the existing list unchanged.\n"
            "- Output ONLY the bullet list. No preamble, no headings."
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        text = (resp.choices[0].message.content or "").strip()
        merged = [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
        merged = [m for m in merged if m]
        return merged

    def _write(self, facts: list[str]) -> None:
        body = "\n".join(f"- {f}" for f in facts)
        self.path.write_text(HEADER + body + "\n", encoding="utf-8")

    # --- pending (noted, unverified) ---
    def _pending_facts(self) -> list[str]:
        return _read_bullets(pending_path())

    def add_pending(self, statement: str, merge: bool = False) -> str:
        """Note a fact into the pending file (unverified)."""
        statement = statement.strip()
        if not statement:
            return ""
        p = pending_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = self._pending_facts()
        if statement in existing:
            return statement
        if merge and self._client is not None and existing:
            try:
                merged = self._merge(existing, statement)
                self._write_pending(merged)
                return merged[-1]
            except Exception:
                pass
        existing.append(statement)
        self._write_pending(existing)
        return statement

    def _write_pending(self, facts: list[str]) -> None:
        p = pending_path()
        body = "\n".join(f"- {f}" for f in facts)
        p.write_text(PENDING_HEADER + body + "\n", encoding="utf-8")

    def pending(self) -> list[str]:
        return self._pending_facts()

    def promote(self, fact: str) -> bool:
        """Move one verified pending fact into learned memories; remove it from pending."""
        fact = fact.strip()
        if fact not in self._pending_facts():
            return False
        self.add(fact, merge=False)
        remaining = [f for f in self._pending_facts() if f != fact]
        self._write_pending(remaining)
        return True

    def promote_all(self) -> int:
        """Auto-keep: move live + pending facts into learned memories."""
        n = 0
        for fact in list(self.live_facts()) + list(self._pending_facts()):
            if not fact:
                continue
            if fact not in self.facts():
                self.add(fact, merge=False)
                n += 1
        self.reset_live()
        self._write_pending([])
        return n

    def discard(self, fact: str) -> bool:
        fact = fact.strip()
        current = self._pending_facts()
        remaining = [f for f in current if f != fact]
        self._write_pending(remaining)
        return len(remaining) < len(current)

    def discard_all(self) -> None:
        self._write_pending([])


TRIGGERS = (
    "గుర్తు పెట్టుకో",
    "గుర్తుపెట్టుకో",
    "గుర్తు పెట్టుకోండి",
    "జ్ఞాపకం పెట్టుకో",
    "remember that",
    "remember this",
    "note that",
)


def find_trigger(text: str) -> str | None:
    low = text.lower()
    for t in TRIGGERS:
        if t.lower() in low:
            return t
    return None


def log_path() -> Path:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return config.LOGS_DIR / f"call_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"


class CallLog:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else log_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, role: str, text: str, **extra) -> None:
        rec = {"ts": round(time.time(), 3), "role": role, "text": text}
        rec.update(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
