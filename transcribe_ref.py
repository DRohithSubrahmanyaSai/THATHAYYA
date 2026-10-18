import sys
from pathlib import Path

from faster_whisper import WhisperModel

BASE = Path(__file__).parent
AUDIO = BASE / "data" / "recordings" / "Thathayya.MP3"
OUT = BASE / "data" / "ref_transcript.txt"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if not AUDIO.exists():
    print(f"ERROR: audio not found: {AUDIO}")
    sys.exit(1)

print("Loading faster-whisper large-v3 (Telugu) on CPU...")
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

print(f"Transcribing {AUDIO.name} ...")
segments, info = model.transcribe(
    str(AUDIO),
    language="te",
    beam_size=5,
    vad_filter=True,
)

text = " ".join(seg.text.strip() for seg in segments).strip()

print("\n--- DRAFT TRANSCRIPT (Telugu) ---")
print(text)
print("--- END ---\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(text + "\n", encoding="utf-8")
print(f"Saved draft to {OUT}")
print("NEXT: correct this by hand with a family member, then save it back to the same file.")
