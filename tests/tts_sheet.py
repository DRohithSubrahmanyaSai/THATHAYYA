"""10-sentence Cartesia voice sheet for Thathayya.

Renders a fixed set of Telugu test sentences with the current cloned voice at
two speeds (0.9 and 1.0) and saves WAVs to tests/out/ plus a scoring checklist.
You listen and pick the best speed/model, then we lock it into .env.

Sentences exercise: name, village/mandal/district, family words, manchi/chaa,
and West Godavari dialect flavor.

Run:
    $env:PATH="$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
    .\.venv\Scripts\python.exe tests/tts_sheet.py
"""
import base64
import asyncio
import json as _json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import soundfile as sf
import websockets

from agent import config

# 10 Telugu test sentences (West Godavari flavor, manchi/chaa sprinkled).
SENTENCES = [
    "నమస్కారం బాబూ, నేనూ తాతయ్య. బాగునాను బాబూ.",
    "నా గ్రామం జిన్నూరు, మండలం పోడూరు, జిల్లా పశ్చిమ గోదావరి.",
    "నా కూతురు నాగమణి, మనీ అంటారు బాబూ.",
    "నా భార్య వరలక్ష్మి, చాలా మంచి వారు బాబూ.",
    "నా మనవడు రోహిత్ సుబ్రహ్మణ్య సాయి, నా మనవాలు ధనలక్ష్మీ.",
    "నా అల్లుడు దగ్గూ సత్యనారాయణ, చాలా బాగురు బాబూ.",
    "నా కొడుకు భవనీ ప్రసాద్, చాలా మంచి వ్యక్తి బాబూ.",
    "చాలా మంచి బాబూ, మీతో మాట్లాడడం చాలా సంతోషం.",
    "చాలు బాబూ, ఇప్పుడు చాయ్ కీయుదాం. మీకు చాయి కావాలా?",
    "లక్ష్మీ, మీ ఆరోగ్యం ఎలా ఉంది బాబూ? చాలా మనది బాబూ.",
]

CASES = [
    ("s09", config.CARTESIA_VOICE_ID, "sonic-3.6", 0.9),
    ("s10", config.CARTESIA_VOICE_ID, "sonic-3.6", 1.0),
]


async def render(text: str, tag: str, voice_id: str, model_id: str, speed: float) -> Path:
    url = f"wss://api.cartesia.ai/tts/websocket?cartesia_version={config.CARTESIA_VERSION}"
    headers = {
        "X-API-Key": config.CARTESIA_API_KEY,
        "Cartesia-Version": config.CARTESIA_VERSION,
    }
    payload = {
        "model_id": model_id,
        "transcript": text,
        "voice": voice_id,
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
        "context_id": "sheet",
        "continue": False,
        "add_timestamps": False,
        "speed": speed,
        "scale": 1.0,
        "temperature": 0.8,
    }
    pcm = b""
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        await ws.send(_json.dumps(payload))
        async for message in ws:
            data = _json.loads(message)
            if data.get("type") == "chunk":
                pcm += base64.b64decode(data.get("data") or "")
            elif data.get("type") == "done":
                break
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    out_path = Path(__file__).resolve().parent / "out" / f"sheet_{tag}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, 24000, format="WAV", subtype="PCM_16")
    return out_path


async def main_async() -> None:
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    checklines = []
    for i, text in enumerate(SENTENCES, start=1):
        tag = f"{i:02d}"
        print(f"[{i:02d}/10] rendering ...")
        for tag2, voice_id, model_id, speed in CASES:
            await render(text, f"{tag}_{tag2}", voice_id, model_id, speed)
            checklines.append(
                f"- [ ] sheet_{tag}_{tag2}.wav  —  speed {speed}  —  \"{text}\""
            )
    checklist = out_dir / "sheet_checklist.md"
    checklist.write_text(
        "# Voice sheet — Thathayya (Cartesia cloned voice)\n\n"
        "Listen to each WAV and score 1-5 on: (a) naturalness, (b) Telugu "
        "pronunciation, (c) West Godavari dialect flavor, (d) speed comfort.\n\n"
        "What to check: జిన్నూరు, కొంపటల/కోంపటల, నాగమణి, manchi, chaa (చాయ్), "
        "వరలక్ష్మి, రోహిత్, ధనలక్ష్మీ, భవనీ.\n\n"
        + "\n".join(checklines)
        + "\n\nPick the best speed (0.9 vs 1.0) and report it; we'll lock it into "
        ".env (CARTESIA_SPEED).\n",
        encoding="utf-8",
    )
    print("Done. WAVs + sheet_checklist.md in tests/out/.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
