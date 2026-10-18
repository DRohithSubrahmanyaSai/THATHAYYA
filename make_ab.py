import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: F401
import numpy as np
import soundfile as sf
import torchaudio

OUT_SR = 24000
GAP = 0.7


def load24k(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != OUT_SR:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, OUT_SR).numpy()
    return a


def main():
    items = [
        ("01_OLD_rawref_forward", r"D:\Thathayya_clone\output_local.wav"),
        ("02_cfg1.5_cleaned", r"D:\Thathayya_clone\data\refs\ab1\cfg1.5_nfe32_seed0.wav"),
        ("03_cfg2.0_cleaned", r"D:\Thathayya_clone\data\refs\ab1\cfg2.0_nfe32_seed0.wav"),
        ("04_cfg2.5_cleaned", r"D:\Thathayya_clone\data\refs\ab1\cfg2.5_nfe32_seed0.wav"),
        ("05_cfg3.0_cleaned", r"D:\Thathayya_clone\data\refs\ab1\cfg3.0_nfe32_seed0.wav"),
    ]
    out_dir = r"D:\Thathayya_clone\data\refs\ab1"
    os.makedirs(out_dir, exist_ok=True)
    silence = np.zeros(int(GAP * OUT_SR), dtype=np.float32)

    parts = []
    lines = []
    for name, path in items:
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue
        a = load24k(path)
        parts.append(a)
        parts.append(silence)
        lines.append(f"{name}  ->  {path}  ({len(a)/OUT_SR:.2f}s)")
        sf.write(os.path.join(out_dir, name + ".wav"), a, OUT_SR)

    merged = np.concatenate(parts).astype(np.float32)
    merged_path = os.path.join(out_dir, "COMPARE.wav")
    sf.write(merged_path, merged, OUT_SR)
    print("\n".join(lines))
    print(f"\nCOMPARE: {merged_path}  ({len(merged)/OUT_SR:.2f}s), {GAP}s gaps, order top->bottom")


if __name__ == "__main__":
    main()
