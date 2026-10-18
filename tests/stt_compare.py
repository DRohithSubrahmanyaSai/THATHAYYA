import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from agent import config
from agent.stt import DeepgramSTT


def load_wav(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def record(duration, sr=16000):
    import sounddevice as sd

    print(f"Recording {duration}s...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten(), sr


def main():
    p = argparse.ArgumentParser(description="Compare STT engines/options on the same clips")
    p.add_argument("--wav", help="Test a WAV file instead of recording")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--clips", type=int, default=3)
    p.add_argument("--whisper", action="store_true", help="Also run faster-whisper large-v3 (slow on CPU)")
    args = p.parse_args()

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    if args.wav:
        clips.append((args.wav, *load_wav(args.wav)))
    else:
        for i in range(args.clips):
            input(f"[{i + 1}/{args.clips}] Press Enter, then speak for {args.duration}s...")
            audio, sr = record(args.duration)
            path = out_dir / f"clip_{time.strftime('%H%M%S')}_{i}.wav"
            sf.write(str(path), audio, sr)
            clips.append((str(path), audio, sr))

    engines = [
        ("deepgram+key+smart", DeepgramSTT(use_keyterms=True, smart_format=True)),
        ("deepgram no-key", DeepgramSTT(use_keyterms=False, smart_format=True)),
        ("deepgram no-smart", DeepgramSTT(use_keyterms=True, smart_format=False)),
    ]
    if args.whisper:
        from agent.stt import FasterWhisperSTT

        engines.append(("whisper-large-v3", FasterWhisperSTT(model_size="large-v3", device="cpu")))

    for path, audio, sr in clips:
        print("\n" + "=" * 72)
        print(f"clip: {path}  ({len(audio) / sr:.1f}s)")
        for label, eng in engines:
            t = time.time()
            try:
                text, conf = eng.transcribe_with_meta(audio, sr)
                c = f"{conf:.2f}" if conf is not None else "n/a"
                print(f"  {label:20s} ({time.time() - t:5.2f}s conf={c:>4}): {text}")
            except Exception as e:  # noqa: BLE001
                print(f"  {label:20s} ERROR: {e}")


if __name__ == "__main__":
    main()
