import json
import os
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

PERSONA_DIR = BASE / "persona"
RUNTIME_DIR = BASE / "runtime"
DATA_DIR = BASE / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
MEMORY_FILE = RUNTIME_DIR / "learned_memories.md"
PENDING_MEMORIES = RUNTIME_DIR / "pending_memories.md"
KEYTERMS_FILE = PERSONA_DIR / "keyterms.txt"
LOGS_DIR = RUNTIME_DIR / "logs"


def env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def load_voice_config() -> dict:
    with open(RUNTIME_DIR / "voice.json", encoding="utf-8") as f:
        return json.load(f)


def load_persona() -> dict:
    return {name: _read(PERSONA_DIR / f"{name}.md") for name in ("system_prompt", "memories", "phrases", "dialect")}


def load_dataset() -> dict:
    path = PERSONA_DIR / "dataset.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render_dataset(ds: dict) -> str:
    """Render the structured dataset into a compact prompt section."""
    if not ds:
        return ""
    lines = ["# About Thathayya (real facts)"]
    ident = ds.get("identity", {})
    if ident:
        lines.append(f"- Name: {ident.get('name', '')}")
        if ident.get("village"):
            lines.append(f"- Lives: {ident.get('village')} village, {ident.get('mandal')} mandal, {ident.get('district')}")
    fam = ds.get("family", {})
    if fam:
        wife = fam.get("wife", {})
        if wife.get("full_name"):
            lines.append(f"- Wife: {wife['full_name']}")
        son = fam.get("son", {})
        if son.get("full_name") or son.get("called"):
            lines.append(f"- Son: {son.get('full_name', '')} (called {son.get('called', '')})")
        dau = fam.get("daughter", {})
        if dau.get("full_name") or dau.get("called"):
            lines.append(f"- Daughter: {dau.get('full_name', '')} (called {dau.get('called', '')})")
        sil = fam.get("son_in_law", {})
        if sil.get("full_name") or sil.get("called"):
            lines.append(f"- Son-in-law: {sil.get('full_name', '')} (called {sil.get('called', '')})")
        sw = fam.get("sons_wife", {})
        if sw.get("full_name"):
            lines.append(f"- Son's wife: {sw['full_name']}")
        for kid in fam.get("grandchildren_via_daughter", []):
            label = kid.get("called") or kid.get("full_name")
            if label:
                lines.append(f"- Grandchild (daughter's side): {label}")
        for kid in fam.get("grandchildren_via_son", []):
            label = kid.get("called") or kid.get("full_name")
            if label:
                lines.append(f"- Grandchild (son's side): {label}")
    work = ds.get("work", {})
    if work:
        lines.append(f"- Job: {work.get('job', '')}" + (f", monthly salary {work['monthly_salary']}" if work.get("monthly_salary") else ""))
    traits = ds.get("personality", {}).get("traits", [])
    if traits:
        lines.append(f"- Personality: {', '.join(traits)}")
    for habit in ds.get("speech_habits", []):
        lines.append(f"- {habit}")
    return "\n".join(lines)


VOICE = load_voice_config()
PERSONA = load_persona()
DATASET = load_dataset()
DATASET_PROMPT = render_dataset(DATASET)

GROQ_API_KEY = env("GROQ_API_KEY")
GROQ_MODEL = env("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MERGE_MODEL = env("GROQ_MERGE_MODEL", GROQ_MODEL)
GROQ_TEMPERATURE = float(env("GROQ_TEMPERATURE", "0.35"))
GROQ_MAX_TOKENS = int(env("GROQ_MAX_TOKENS", "120"))
GROQ_REASONING_EFFORT = env("GROQ_REASONING_EFFORT", "low")
GROQ_TIMEOUT_S = float(env("GROQ_TIMEOUT_S", "8"))
GROQ_FALLBACK_MODEL = env("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
GROQ_LEARN_MODEL = env("GROQ_LEARN_MODEL", "openai/gpt-oss-20b")
LEARN_MIN_CONF = float(env("LEARN_MIN_CONF", "0.0"))
CARTESIA_SPEED = float(env("CARTESIA_SPEED", "0.9"))
CARTESIA_SCALE = float(env("CARTESIA_SCALE", "1.0"))
CARTESIA_TEMPERATURE = float(env("CARTESIA_TEMPERATURE", "0.8"))
CARTESIA_API_KEY = env("CARTESIA_API_KEY")
CARTESIA_VOICE_ID = env("CARTESIA_VOICE_ID")
CARTESIA_MODEL = env("CARTESIA_MODEL", VOICE.get("cartesia", {}).get("model", "sonic-3.6"))
CARTESIA_VERSION = env("CARTESIA_VERSION", "2026-08-14")
DEEPGRAM_API_KEY = env("DEEPGRAM_API_KEY")
HF_TOKEN = env("HF_TOKEN")

STT_ENGINE = env("STT_ENGINE", "deepgram")
TTS_LIVE_ENGINE = env("TTS_LIVE_ENGINE", VOICE.get("engines", {}).get("live", "cartesia"))
TTS_DRAFT_ENGINE = env("TTS_DRAFT_ENGINE", VOICE.get("engines", {}).get("draft", "indicf5_local"))
LLM_ENGINE = env("LLM_ENGINE", "groq")

AGENT_MODE = env("AGENT_MODE", "stream")

DEEPGRAM_MODEL = env("DEEPGRAM_MODEL", "nova-3")
STT_LANGUAGE = env("STT_LANGUAGE", "te")
STT_CONFIDENCE_MIN = float(env("STT_CONFIDENCE_MIN", "0.6"))

DG_ENDPOINTING_MS = int(env("DG_ENDPOINTING_MS", "350"))
DG_UTTERANCE_END_MS = int(env("DG_UTTERANCE_END_MS", "1000"))
DG_KEEPALIVE_S = float(env("DG_KEEPALIVE_S", "5"))
MAX_SILENCE_MS = int(env("MAX_SILENCE_MS", "600"))
LISTEN_PREFIX_S = float(env("LISTEN_PREFIX_S", "0.25"))
VAD_THRESHOLD = float(env("VAD_THRESHOLD", "0.5"))
BARGE_IN = env("BARGE_IN", "1") == "1"
BARGE_IN_RMS = float(env("BARGE_IN_RMS", "0.025"))
BARGE_IN_MIN_S = float(env("BARGE_IN_MIN_S", "0.15"))

CARTESIA_SR = int(env("CARTESIA_SR", "24000"))
PLAYER_SR = int(env("PLAYER_SR", "24000"))

LOW_CONFIDENCE_REPLY = env("LOW_CONFIDENCE_REPLY", "ఏమన్నావ్ బాబూ, క్లియర్‌గా చెప్పు.")
