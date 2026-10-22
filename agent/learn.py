"""Hybrid live-learning for Thathayya.

After each user turn, decide if the user just taught a durable fact
(name, place, preference, or a correction of an old fact) and, if so,
write it into the live-memory layer immediately (so the next reply can use
it) and into the pending file for review. Explicit teaching phrases
("గుర్తు పెట్టుకో") are always learned.
"""

import re

from . import config
from .memory import Memory, find_trigger

# A turn must look like it carries a fact before we pay for an LLM call.
# Use short roots so the pre-filter is robust to STT spelling variation.
_FACT_HINTS = (
    "పేర", "జనమ", "ఆగం", "గ్రామ", "మండల", "ఊర",
    "వైఫ్", "భార్య", "కొడ", "మరద", "నాన", "అత్త", "మామ", "బావ",
    "పెసన", "గురత", "పురికొం", "తెలపా", "ఇల్ల", "పొల", "పని", "చెట్ట",
    "అల్లు", "మనవ", "కూతుర", "కాద", "అనాల", "అది",
    "remember", "note that", "my name", "lives",
)

# Match ASCII (English/meta) content that must never reach TTS.
_ASCII_BLOCK = re.compile(r"[A-Za-z]{2,}")
_ASCII_WORDS = re.compile(r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*")


def _strip_trigger(text: str) -> str:
    trig = find_trigger(text)
    if trig:
        text = text.replace(trig, "", 1)
    return text.strip(" ,.!?।")


def _looks_like_fact(text: str) -> bool:
    low = text.lower()
    return any(h.lower() in low for h in _FACT_HINTS)


class LiveLearner:
    def __init__(self, memory: Memory, client, model: str | None = None):
        self.memory = memory
        self.client = client
        self.model = model or config.GROQ_LEARN_MODEL

    def maybe_learn(self, user_text: str, conf: float | None = None) -> str:
        """Note a fact into live + pending memory. Returns the learned fact or ""."""
        user_text = user_text.strip()
        if not user_text:
            return ""

        if find_trigger(user_text):
            fact = _strip_trigger(user_text)
            self.memory.add_live(fact)
            self.memory.add_pending(fact, merge=False)
            return fact

        if not _looks_like_fact(user_text):
            return ""

        try:
            fact = self._extract_fact(user_text)
        except Exception as e:  # noqa: BLE001
            print(f"[learn error] {e}")
            return ""

        if not fact:
            return ""
        self.memory.add_live(fact)
        self.memory.add_pending(fact, merge=False)
        return fact

    def _extract_fact(self, user_text: str) -> str | None:
        existing = self.memory.facts() + self.memory.live_facts()
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
            max_tokens=150,
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
        text = re.sub(r"\([^)]*\)", "", text)
        cleaned = []
        for frag in re.split(r"[.?!।\n]+", text):
            frag = _ASCII_WORDS.sub("", frag).strip(" ,;:-")
            if frag:
                cleaned.append(frag)
        out = " ".join(cleaned).strip()
        out = re.sub(r"(.+?)\.\s*\1\.", r"\1.", out)
        return out
