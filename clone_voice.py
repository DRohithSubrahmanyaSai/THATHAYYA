import os
import sys
import json
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

ENV_PATH = Path(__file__).parent / ".env"
DEFAULT_CLIP = Path(__file__).parent / "Thathayya.MP3"

clip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLIP
voice_name = sys.argv[2] if len(sys.argv) > 2 else "Thathayya"

if not clip_path.exists():
    print(f"ERROR: Audio file not found: {clip_path}")
    sys.exit(1)

size_mb = clip_path.stat().st_size / (1024 * 1024)
if size_mb > 16:
    print(f"ERROR: File is {size_mb:.1f} MB. Cartesia's limit is 16 MB.")
    sys.exit(1)

url = "https://api.cartesia.ai/voices/clone"

headers = {
    "X-API-Key": CARTESIA_API_KEY,
    "Cartesia-Version": "2026-08-14",
}

data = {
    "name": voice_name,
    "language": "te",
    "access": "private",
}

print(f"Cloning voice '{voice_name}' from {clip_path.name} ({size_mb:.2f} MB)...")

try:
    with open(clip_path, "rb") as f:
        files = {"clip": (clip_path.name, f, "audio/mpeg")}
        response = requests.post(url, headers=headers, data=data, files=files, timeout=120)
except requests.exceptions.ConnectionError:
    print("ERROR: Could not connect to api.cartesia.ai. Check your internet.")
    sys.exit(1)
except requests.exceptions.Timeout:
    print("ERROR: Request timed out after 120 seconds.")
    sys.exit(1)

if response.status_code != 200:
    print(f"ERROR: API returned {response.status_code}")
    print(response.text)
    sys.exit(1)

try:
    meta = response.json()
except ValueError:
    print("ERROR: Response was not JSON.")
    print(response.text)
    sys.exit(1)

voice_id = meta.get("id")
print("SUCCESS: Voice cloned.")
print(json.dumps(meta, indent=2, ensure_ascii=False))

if not voice_id:
    print("ERROR: No voice id returned.")
    sys.exit(1)

def set_env_var(key, value):
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(key + "="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

set_env_var("CARTESIA_VOICE_ID", voice_id)
print(f"Saved CARTESIA_VOICE_ID={voice_id} to {ENV_PATH}")
