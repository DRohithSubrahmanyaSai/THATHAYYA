import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse

import torch  # noqa: F401
import numpy as np
import soundfile as sf
import torchaudio

OUT_SR = 24000


def load24k(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != OUT_SR:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, OUT_SR).numpy()
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--gap", type=float, default=0.7)
    args = p.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    silence = np.zeros(int(args.gap * OUT_SR), dtype=np.float32)
    parts = []
    for f in args.files:
        a = load24k(f)
        parts.append(a)
        parts.append(silence)
        print(f"  + {os.path.basename(f):40s} {len(a)/OUT_SR:5.2f}s")
    merged = np.concatenate(parts).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, merged, OUT_SR)
    print(f"\nCOMPARE: {args.out}  ({len(merged)/OUT_SR:.2f}s)")
    print("order:")
    for i, lab in enumerate(labels, 1):
        print(f"  {i:02d} {lab}")


if __name__ == "__main__":
    main()
