import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
if not CARTESIA_API_KEY or "*" in CARTESIA_API_KEY:
    print("ERROR: Set your real CARTESIA_API_KEY in .env before running.")
    sys.exit(1)

url = "https://api.cartesia.ai/tts/bytes"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": CARTESIA_API_KEY,
    "Cartesia-Version": "2026-08-14",
}

DEFAULT_TRANSCRIPT = (
    "నమస్కారం! ఎలా ఉన్నావు? తిన్నావా? "
    "సంతోషంగా ఉండు. నేను నిన్ను చూస్తున్నాను."
)

transcript = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRANSCRIPT
output_name = sys.argv[2] if len(sys.argv) > 2 else "output.wav"

payload = {
    "model_id": "sonic-3.6",
    "transcript": transcript,
    "voice": "bdaae173-3796-4f4d-9ba1-cc94c7de90c8",
    "output_format": {
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_rate": 44100,
    },
    "generation_config": {
        "speed": 1,
        "volume": 1,
    },
}

print("Sending TTS request to Cartesia...")

try:
    response = requests.post(url, json=payload, headers=headers, timeout=30)
except requests.exceptions.ConnectionError:
    print("ERROR: Could not connect to api.cartesia.ai. Check your internet.")
    sys.exit(1)
except requests.exceptions.Timeout:
    print("ERROR: Request timed out after 30 seconds.")
    sys.exit(1)

if response.status_code != 200:
    print(f"ERROR: API returned {response.status_code}")
    print(response.text)
    sys.exit(1)

audio_path = Path(__file__).parent / output_name
audio_path.write_bytes(response.content)

print(f"Transcript: {transcript}")
print(f"SUCCESS: Audio saved to {audio_path}")
print(f"File size: {len(response.content)} bytes")
