import json
import time
from pathlib import Path

from . import config

HEADER = (
    "# Learned memories — Thathayya\n\n"
    "> Facts taught to Thathayya during conversations. Auto-managed; safe to edit by hand.\n\n"
)


class Memory:
    def __init__(self, path: Path | None = None, client=None, model: str | None = None):
        self.path = Path(path) if path else config.MEMORY_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(HEADER, encoding="utf-8")
        self.model = model or config.GROQ_MERGE_MODEL
        self._client = client
        if self._client is None and config.GROQ_API_KEY:
            from groq import Groq

            self._client = Groq(api_key=config.GROQ_API_KEY)

    def facts(self) -> list[str]:
        out: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("- "):
                out.append(line[2:].strip())
        return out

    def count(self) -> int:
        return len(self.facts())

    def as_prompt(self) -> str:
        facts = self.facts()
        if not facts:
            return ""
        return "\n".join(f"- {f}" for f in facts)

    def add(self, statement: str, merge: bool = True) -> str:
        statement = statement.strip()
        if not statement:
            return ""
        existing = self.facts()
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
            "- Merge duplicates; if the new statement corrects an old fact, replace it.\n"
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
