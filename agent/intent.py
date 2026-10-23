"""Fast, local intent routing for the live call.

Keeps the voice loop responsive: no LLM call is made here. The goal is to
catch the cases that caused rambling / hallucinated replies:
- unclear audio (low STT confidence)
- a pronunciation/word correction from the client
- a "who is X" question about someone we do not know
- explicit teaching
"""

import re

from . import config
from .memory import find_trigger

_CORRECTION_NEG = (
    "కాదు", "కాదండి", "అనకూడదు", "అనరు", "తప్పు", "సరిగ్గా", "లేదు",
)
_CORRECTION_FIX = (
    "అనాలి", "అనండి", "అంటారు", "చెప్ప", "అలా", "అని", "అనే",
    "అది", "say", "pronounce", "not",
)

# Tokens that showed up as STT noise in real calls. Asking "what did you say?"
# is better than letting the LLM invent a meaning for them.
_ASR_GARBLE = ("గోరేంటి", "గోరెండు", "గోరెండ్", "గోరెండ", "గోరు")

_GREETINGS = (
    "హలో", "హాయ్", "నమస్కారం", "నమస్తే", "బాగున్నావా", "ఏం చేస్తున్నావ్",
    "hello", "hi", "hey",
)

_PERSON_Q = ("ఎవరు", "ఎవరి", "ఎవరికి", "ఎవరైనా", "who")

# The client is not correcting one word; they are saying the whole answer was
# wrong / invented / nonsensical. Repair instead of continuing normally.
_COMPLAINT = (
    "అసలు ఉండదు", "ఉండదు కద", "పిచ్చి", "తప్పు", "అబద్ధం", "నాన్సెన్స్",
    "వట్టి మాటలు", "అర్థం కావడం లేదు", "అర్థం కాలేదు", "ఏం మాటలాడుతున్న",
    "అలా కాదు", "సరిగ్గా లేదు", "బాగా లేదు", "నిజం కాదు", "తల తినేస్తున్న",
)

# Only a bare request to repeat, not "అదే అండి, అన్నం తిన్నారా?".
_REPEAT_SHAPE = re.compile(
    r"^\s*(?:అదే|అదేంటి|అదే ఏంటి|మళ్ళీ|మళ్ళీ చెప్పు|మళ్ళీ చెప్పండి|"
    r"ఏం అన్నారు|ఏం అన్నావ్|అర్థం కాలేదు|అర్థం కావడం లేదు|క్లియర్‌గా చెప్పండి)"
    r"(?:\s+బాబూ|\s+అండి)?\s*[?.!]*$"
)

# Words that identify a topic, not a person/name. They must never make an
# "evaru ..." question look known.
_GENERIC_TOKENS = {
    "పేరు", "పేరేంటి", "ఊరు", "ఇల్లు", "ఇంటి",
    "గ్రామం", "కుటుంబం", "వాళ్ళు", "గారు",
    "అని", "కాదు", "అవును", "ఏంటి", "ఎవరు",
    "మా", "నా", "మీ", "మన", "అది", "ఇది", "బాబూ", "అండి",
}

_CORRECTION_SHAPE = re.compile(
    r"\S+\s+(?:"
    + "|".join(re.escape(k) for k in sorted(_CORRECTION_NEG, key=len, reverse=True))
    + r")\s+(?!బాబూ|అండి|గారు)[^\s,.!?]{2,}"
)

_RELATION_WORDS = (
    "కూతురు", "కూతురి", "కొడుకు", "కొడుకు", "అల్లుడు", "అల్లుడి",
    "భార్య", "భర్త", "మనవడు", "మనవరాలు", "నాన్న", "అమ్మ", "తాతయ్య",
    "అమ్మమ్మ", "నానమ్మ", "పేరు", "పేరేంటి", "ఇల్లు", "ఊరు", "గ్రామం",
    "కుటుంబం", "ఇంటి", "వాళ్ళు", "గారు",
)


def _known_tokens() -> set[str]:
    toks: set[str] = {w for w in _RELATION_WORDS if w not in _GENERIC_TOKENS}
    ds = getattr(config, "DATASET", {}) or {}

    def add(value) -> None:
        if not isinstance(value, str):
            return
        value = value.strip().lower()
        if not value:
            return
        if value not in _GENERIC_TOKENS:
            toks.add(value)
        for part in value.replace("(", " ").replace(")", " ").split():
            if len(part) >= 2 and part not in _GENERIC_TOKENS:
                toks.add(part)

    ident = ds.get("identity", {}) or {}
    for key in ("name", "village", "mandal", "district"):
        add(ident.get(key))
    fam = ds.get("family", {}) or {}
    for value in fam.values():
        if isinstance(value, dict):
            add(value.get("full_name"))
            add(value.get("called"))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    add(item.get("full_name"))
                    add(item.get("called"))

    for path in (config.MEMORY_FILE, config.LIVE_MEMORY_FILE):
        try:
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("- "):
                        for part in line[2:].split():
                            part = part.strip().lower()
                            if len(part) >= 2 and part not in _GENERIC_TOKENS:
                                toks.add(part)
        except OSError:
            pass
    return toks


def is_correction(text: str) -> bool:
    low = text.lower()
    if not any(k in low for k in _CORRECTION_NEG):
        return False
    if any(k in low for k in _CORRECTION_FIX):
        return True
    return bool(_CORRECTION_SHAPE.search(text))


def is_complaint(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in _COMPLAINT)


def is_repeat_request(text: str) -> bool:
    return bool(_REPEAT_SHAPE.match((text or "").strip()))


def is_greeting(text: str) -> bool:
    low = text.lower().strip()
    if len(low.split()) > 6:
        return False
    return any(g in low for g in _GREETINGS)


def unknown_person_question(text: str) -> bool:
    low = text.lower()
    if len(text.split()) < 3:
        return False
    if not any(q in low for q in _PERSON_Q):
        return False
    known = _known_tokens()
    return not any(k and k in low for k in known)


def classify(text: str, conf: float | None = None) -> str:
    """Return one of: unclear, correction, complaint, teach, unknown_person,
    repeat, greeting, normal."""
    text = (text or "").strip()
    if conf is not None and conf < config.STT_CONFIDENCE_MIN:
        return "unclear"
    if not text:
        return "unclear"
    if is_correction(text):
        return "correction"
    if is_complaint(text):
        return "complaint"
    if any(g in text for g in _ASR_GARBLE):
        return "unclear"
    if find_trigger(text):
        return "teach"
    if unknown_person_question(text):
        return "unknown_person"
    if is_repeat_request(text):
        return "repeat"
    if is_greeting(text):
        return "greeting"
    return "normal"
