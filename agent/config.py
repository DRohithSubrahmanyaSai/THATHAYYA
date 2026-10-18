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


VOICE = load_voice_config()
PERSONA = load_persona()

GROQ_API_KEY = env("GROQ_API_KEY")
GROQ_MODEL = env("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MERGE_MODEL = env("GROQ_MERGE_MODEL", GROQ_MODEL)
CARTESIA_API_KEY = env("CARTESIA_API_KEY")
CARTESIA_VOICE_ID = env("CARTESIA_VOICE_ID")
CARTESIA_MODEL = env("CARTESIA_MODEL", VOICE.get("cartesia", {}).get("model", "sonic-3.6"))
DEEPGRAM_API_KEY = env("DEEPGRAM_API_KEY")
HF_TOKEN = env("HF_TOKEN")

STT_ENGINE = env("STT_ENGINE", "deepgram")
TTS_LIVE_ENGINE = env("TTS_LIVE_ENGINE", VOICE.get("engines", {}).get("live", "cartesia"))
TTS_DRAFT_ENGINE = env("TTS_DRAFT_ENGINE", VOICE.get("engines", {}).get("draft", "indicf5_local"))
LLM_ENGINE = env("LLM_ENGINE", "groq")
