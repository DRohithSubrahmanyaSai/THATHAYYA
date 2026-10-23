"""A/B test Cartesia TTS pronunciation for Telugu.

Renders a fixed Telugu line with several speed/sample-rate/dictionary combos and
saves WAVs to tests/out/ so you can listen and pick the best.

Run:
    $env:PYTHONIOENCODING='utf-8'
    .\.venv\Scripts\python.exe tests/tts_ab.py
"""
import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import soundfile as sf
import websockets

from agent import config

TEST_LINE = (
    "నమస్కారం బాబూ, నేను సత్యనారాయణ. మా ఊరు జిన్నూరు, పొదూరు మండలం. "
    "ఇంట్లో అన్నం, కూర తిన్నాను."
)

# (label, speed, sample_rate, use_dictionary)
CASES = [
    ("dict on  24k speed 0.9", 0.9, 24000, True),
    ("dict on  24k speed 1.0", 1.0, 24000, True),
    ("dict on  44k speed 0.9", 0.9, 44100, True),
    ("dict off 24k speed 0.9", 0.9, 24000, False),
]


def synthesize(speed: float, sample_rate: int, use_dictionary: bool, out_path: Path) -> None:
    url = f"wss://api.cartesia.ai/tts/websocket?cartesia_version={config.CARTESIA_VERSION}"
    headers = {
        "X-API-Key": config.CARTESIA_API_KEY,
        "Cartesia-Version": config.CARTESIA_VERSION,
    }

    async def run() -> None:
        async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
            ctx = "ab_test"
            payload = {
                "model_id": config.CARTESIA_MODEL,
                "transcript": TEST_LINE,
                "voice": config.CARTESIA_VOICE_ID,
                "language": config.CARTESIA_LANGUAGE,
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": sample_rate,
                },
                "context_id": ctx,
                "continue": False,
                "add_timestamps": False,
                "max_buffer_delay_ms": config.CARTESIA_MAX_BUFFER_MS,
                "generation_config": {"speed": speed},
            }
            if use_dictionary and config.CARTESIA_PRON_DICT_ID:
                payload["pronunciation_dict_id"] = config.CARTESIA_PRON_DICT_ID
            await ws.send(json.dumps(payload))
            pcm = b""
            async for message in ws:
                data = json.loads(message)
                kind = data.get("type")
                if kind == "chunk":
                    pcm += base64.b64decode(data.get("data") or "")
                elif kind == "done":
                    break
                elif kind == "error":
                    raise RuntimeError(str(data)[:200])
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio, sample_rate, format="WAV", subtype="PCM_16")

    asyncio.run(run())


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "out"
    for label, speed, sample_rate, use_dict in CASES:
        tag = f"s{speed}_{sample_rate // 1000}k_{'dict' if use_dict else 'nodict'}"
        out_path = out_dir / f"tts_ab_{tag}.wav"
        print(f"[{label}] -> {out_path.name}")
        try:
            synthesize(speed, sample_rate, use_dict, out_path)
        except Exception as e:  # noqa: BLE001
            print(f"  error: {e}")
    print("Done. Listen to the WAVs in tests/out/ and pick the best.")


if __name__ == "__main__":
    main()
