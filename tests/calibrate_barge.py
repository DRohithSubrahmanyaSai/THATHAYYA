import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd

from agent import config


def record(seconds, sr=16000):
    print(f"  recording {seconds}s...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def rms(x):
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def block_rms(audio, block_ms=32, sr=16000):
    n = int(sr * block_ms / 1000)
    return [rms(audio[i : i + n]) for i in range(0, len(audio) - n + 1, n)]


def main():
    print("Barge-in calibration. Stay quiet, then speak normally.\n")
    input("Press Enter, then be SILENT for 3s...")
    silence = record(3.0)
    input("Press Enter, then SPEAK normally for 3s (as if interrupting)...")
    speech = record(3.0)

    s_blocks = block_rms(silence)
    v_blocks = block_rms(speech)
    s_mean, s_max = np.mean(s_blocks), np.max(s_blocks)
    v_mean, v_min = np.mean(v_blocks), np.min(v_blocks)

    print("\n--- results (per-32ms block RMS) ---")
    print(f"silence: mean={s_mean:.4f}  max={s_max:.4f}")
    print(f"speech : mean={v_mean:.4f}  min={v_min:.4f}")

    suggested = round(max(s_max * 2.0, v_mean * 0.5), 4)
    print(f"\nSuggested BARGE_IN_RMS = {suggested}")
    print(f"Add to .env:  BARGE_IN_RMS={suggested}")
    if s_max * 2.0 > v_mean * 0.5:
        print("(Warning: background noise is high relative to your voice; consider headphones.)")


if __name__ == "__main__":
    main()
