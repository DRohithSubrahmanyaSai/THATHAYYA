import os
import sys
from pathlib import Path

# IMPORTANT: import torch FIRST. Importing pandas/pyarrow/datasets before torch
# breaks torch's DLL loading on this Windows setup.
import torch

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

# torchaudio 2.11 removed its native decoders and requires TorchCodec for
# torchaudio.load(). f5_tts only uses torchaudio.load() + Resample() in the
# inference path, so shim load() onto soundfile (already a dependency).
import torchaudio


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _soundfile_load

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ.setdefault("HF_HOME", str(BASE / ".cache" / "hf"))

REF_AUDIO = BASE / "data" / "recordings" / "Thathayya.MP3"
REF_TEXT_FILE = BASE / "data" / "ref_transcript.txt"

DEFAULT_TEXT = (
    "నమస్కారం! ఎలా ఉన్నావు? తిన్నావా? "
    "సంతోషంగా ఉండు. నేను నిన్ను చూస్తున్నాను."
)

transcript = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
output_name = sys.argv[2] if len(sys.argv) > 2 else "output_local.wav"
speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

if not REF_AUDIO.exists():
    print(f"ERROR: reference audio not found: {REF_AUDIO}")
    sys.exit(1)
if not REF_TEXT_FILE.exists():
    print(f"ERROR: reference transcript not found: {REF_TEXT_FILE}")
    print("Run transcribe_ref.py first, then correct the transcript by hand.")
    sys.exit(1)

ref_text = REF_TEXT_FILE.read_text(encoding="utf-8").strip()
if not ref_text:
    print("ERROR: reference transcript is empty.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("Loading IndicF5 (first run downloads ~1.6 GB to .cache/hf)...")

from transformers import AutoModel

model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

try:
    model.to(device)
except Exception as e:
    print(f"note: could not move model to {device}: {e}")

model.config.speed = speed

print(f"Reference audio: {REF_AUDIO.name}")
print(f"Speed          : {speed}")
print(f"Reference text : {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")
print(f"Generating     : {transcript}")

audio = model(transcript, ref_audio_path=str(REF_AUDIO), ref_text=ref_text)

if hasattr(audio, "detach"):
    audio = audio.detach().cpu().numpy()
audio = np.asarray(audio)
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
audio = audio.astype(np.float32)

out = BASE / output_name
sf.write(str(out), audio, 24000)
print(f"SUCCESS: saved {out} ({out.stat().st_size} bytes, {len(audio) / 24000:.1f}s)")
