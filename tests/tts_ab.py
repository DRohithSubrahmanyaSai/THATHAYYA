"""A/B test Cartesia TTS pronunciation for Telugu.

Renders a fixed Telugu line with several voice/model/speed combos and saves
WAVs to tests/out/ so you can listen and pick the best.

Run:
    $env:PATH="$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
    .\.venv\Scripts\python.exe tests/tts_ab.py
"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import soundfile as sf
import websockets
import asyncio

from agent import config

# Fixed Telugu line that exercises several words the voice mispronounces.
TEST_LINE = "నమస్కారం బాబూ, నేనూ జిన్నూరు గ్రామం, పొదూరు మండలం నుండ్డి విసస్థితున్నాం, బాబూ."

# (label, voice_id, model_id, speed)
CASES = [
    ("sonic-3.6 speed 0.9", config.CARTESIA_VOICE_ID, "sonic-3.6", 0.9),
    ("sonic-3.6 speed 1.0", config.CARTESIA_VOICE_ID, "sonic-3.6", 1.0),
    ("sonic-3.5 speed 0.9", config.CARTESIA_VOICE_ID, "sonic-3.5", 0.9),
]


def synthesize(voice_id: str, model_id: str, speed: float, out_path: Path) -> None:
    url = f"wss://api.cartesia.ai/tts/websocket?cartesia_version={config.CARTESIA_VERSION}"
    headers = {
        "X-API-Key": config.CARTESIA_API_KEY,
        "Cartesia-Version": config.CARTESIA_VERSION,
    }

    async def run() -> None:
        async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
            ctx = "ab_test"
            payload = {
                "model_id": model_id,
                "transcript": TEST_LINE,
                "voice": voice_id,
                "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
                "context_id": ctx,
                "continue": False,
                "add_timestamps": False,
                "speed": speed,
                "scale": 1.0,
                "temperature": 0.8,
            }
            await ws.send(__import__("json").dumps(payload))
            pcm = b""
            async for message in ws:
                import json as _json
                data = _json.loads(message)
                kind = data.get("type")
                if kind == "chunk":
                    pcm += base64.b64decode(data.get("data") or "")
                elif kind == "done":
                    break
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio, 24000, format="WAV", subtype="PCM_16")

    asyncio.run(run())


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "out"
    for label, voice_id, model_id, speed in CASES:
        tag = f"{model_id}_s{speed}"
        out_path = out_dir / f"tts_ab_{tag}.wav"
        print(f"[{label}] -> {out_path.name}")
        try:
            synthesize(voice_id, model_id, speed, out_path)
        except Exception as e:
            print(f"  error: {e}")
    print("Done. Listen to the WAVs in tests/out/ and pick the best model/speed.")


if __name__ == "__main__":
    main()
