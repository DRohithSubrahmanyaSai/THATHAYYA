"""Hybrid live-learning for Thathayya.

After each user turn, decide if the user just taught a durable fact
(name, place, preference, or a correction of an old fact) and, if so,
merge it into the learned-memory file via Memory.add().

Explicit teaching phrases ("teach me this" / "గుర్తు పెట్టుకో") are always
learned. Otherwise a cheap LLM extracts only clearly-stated facts, so we
don't spam memory with small talk.
"""

import re

from . import config
from .memory import Memory, find_trigger

# A turn must look like it carries a fact before we pay for an LLM call.
# Use short roots so the pre-filter is robust to STT spelling variation.
_FACT_HINTS = (
    "పేర", "జనమ", "ఆగం", "గ్రామ", "మండల",
    "వైఫ్", "భార్య", "కొడ", "మరద", "నాన",
    "పెసన", "గురత", "పురికొం", "తెలపా",
    "remember", "note that", "my name", "lives",
)

# Match ASCII (English/meta) content that must never reach TTS.
_ASCII_BLOCK = re.compile(r"[A-Za-z]{2,}")


def _looks_like_fact(text: str) -> bool:
    low = text.lower()
    return any(h.lower() in low for h in _FACT_HINTS)


class LiveLearner:
    def __init__(self, memory: Memory, client, model: str | None = None):
        self.memory = memory
        self.client = client
        self.model = model or config.GROQ_LEARN_MODEL

    def maybe_learn(self, user_text: str, conf: float | None = None) -> bool:
        """Learn from a user turn. Returns True if a fact was stored."""
        user_text = user_text.strip()
        if not user_text:
            return False

        # Explicit trigger -> always learn (cheap, no LLM).
        if find_trigger(user_text):
            self.memory.add(user_text)
            return True

        # Implicit capture: only when the turn plausibly states a fact.
        if not _looks_like_fact(user_text):
            return False

        if conf is not None and conf < config.LEARN_MIN_CONF:
            return False

        try:
            fact = self._extract_fact(user_text)
        except Exception as e:  # noqa: BLE001
            print(f"[learn error] {e}")
            return False

        if not fact:
            return False
        self.memory.add(fact)
        return True

    def _extract_fact(self, user_text: str) -> str | None:
        existing = self.memory.facts()
        current = "\n".join(f"- {f}" for f in existing)
        prompt = (
            "You are extracting durable facts about Thathayya from a line his grandson just said.\n"
            f"Existing facts:\n{current}\n\n"
            f'User line: "{user_text}"\n\n'
            "If this line states a NEW durable fact (a name, place, job, preference, or a correction "
            "of an existing fact), return that single fact as one short bullet line starting with '- '. "
            "If it is small talk or repeats an existing fact, return nothing.\n"
            "Write the fact in Telugu (the same language as the existing facts).\n"
            "Output ONLY the bullet line, or the single word 'NONE'."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
            timeout=config.GROQ_TIMEOUT_S,
            reasoning_effort="low",
        )
        out = (resp.choices[0].message.content or "").strip()
        if out.upper() == "NONE":
            return None
        out = out.strip()
        if out.startswith("- "):
            out = out[2:]
        return out or None

    @staticmethod
    def sanitize(text: str) -> str:
        """Strip ASCII/English/meta fragments so only the spoken Telugu line reaches TTS."""
        if not text:
            return ""
        # Drop any line/fragment containing ASCII letters (English instructions, meta notes).
        cleaned = []
        for frag in text.split(".?!।\n"):
            frag = frag.strip()
            if frag and not _ASCII_BLOCK.search(frag):
                cleaned.append(frag)
        out = " ".join(cleaned).strip()
        # Collapse accidental exact-duplicate adjacent sentences ("X. X.")
        out = re.sub(r"(.+?)\.\s*\1\.", r"\1.", out)
        return out
