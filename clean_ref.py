import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse

import torch  # noqa: F401  (import torch first)
import numpy as np
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt


def highpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--no-denoise", action="store_true")
    p.add_argument("--prop-decrease", type=float, default=0.5)
    p.add_argument("--highpass", type=float, default=60.0)
    p.add_argument("--target-lufs", type=float, default=-20.0)
    args = p.parse_args()

    audio, sr = sf.read(args.inp, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"in:  {args.inp}  sr={sr}  dur={len(audio)/sr:.3f}s")

    s = int(args.start * sr)
    e = len(audio) if args.end is None else int(args.end * sr)
    audio = audio[s:e]
    print(f"window: {args.start:.2f}-{args.end if args.end else len(audio)/sr + args.start:.2f}s  -> {len(audio)/sr:.3f}s")

    if args.highpass > 0:
        audio = highpass(audio, sr, args.highpass)
        print(f"highpass: {args.highpass} Hz")

    if not args.no_denoise:
        audio = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=False,
            prop_decrease=args.prop_decrease,
            n_fft=1024,
            win_length=1024,
        ).astype(np.float32)
        print(f"denoise: non-stationary prop_decrease={args.prop_decrease}")

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, args.target_lufs).astype(np.float32)
    print(f"loudness: {loudness:.2f} -> {args.target_lufs:.2f} LUFS")

    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio = (audio / peak * 0.99).astype(np.float32)
        print(f"peak-limited to 0.99 (was {peak:.3f})")

    sf.write(args.out, audio, sr)
    print(f"out: {args.out}  dur={len(audio)/sr:.3f}s  peak={np.max(np.abs(audio)):.3f}  rms={np.sqrt(np.mean(audio**2)):.4f}")


if __name__ == "__main__":
    main()
