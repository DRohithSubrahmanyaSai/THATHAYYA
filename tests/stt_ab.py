"""A/B test Deepgram Telugu accuracy on the reference clip.

Transcribes data/recordings/Thathayya.MP3 under different endpointing
settings and prints the transcript + confidence for each, so you can hear
which setting is most accurate.

Run:
    $env:PATH="$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
    .\.venv\Scripts\python.exe tests/stt_ab.py
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import config
from agent.stt import STT

SETTINGS = [
    ("baseline", {"use_keyterms": False, "smart_format": True}),
    ("with keyterms", {"use_keyterms": True, "smart_format": True}),
    ("keyterms + smart_format off", {"use_keyterms": True, "smart_format": False}),
]


def main() -> None:
    clip = config.RECORDINGS_DIR / "Thathayya.MP3"
    if not clip.exists():
        print(f"Missing reference clip: {clip}")
        return
    audio, sr = sf.read(str(clip), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    for label, overrides in SETTINGS:
        params = {"use_keyterms": True, "smart_format": True}
        params.update(overrides)
        stt = STT("deepgram", **params)
        text, conf = stt.transcribe_with_meta(audio, sr)
        c = f"{conf:.2f}" if conf is not None else "n/a"
        print(f"[{label}] conf={c}\n  {text}\n")


if __name__ == "__main__":
    main()
