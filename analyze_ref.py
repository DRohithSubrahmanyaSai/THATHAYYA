import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: F401  (import torch first: Windows c10.dll load order)
import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt


def main(wav_path: str = r"D:\Thathayya_clone\data\refs\thathayya_raw24k.wav"):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    print(f"file      : {wav_path}")
    print(f"sr        : {sr} Hz")
    print(f"duration  : {dur:.3f} s")
    print(f"samples   : {len(audio)}")
    print(f"peak      : {np.max(np.abs(audio)):.4f}")
    print(f"rms       : {np.sqrt(np.mean(audio ** 2)):.4f}")

    win = int(0.05 * sr)
    hop = int(0.025 * sr)
    frames = 1 + (len(audio) - win) // hop
    rms = np.empty(frames)
    for i in range(frames):
        seg = audio[i * hop : i * hop + win]
        rms[i] = np.sqrt(np.mean(seg ** 2) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    times = np.arange(frames) * hop / sr

    peak_db = db.max()
    thresh = peak_db - 25.0
    speech = db > thresh
    print(f"peak frame dB : {peak_db:.1f}")
    print(f"speech thr    : {thresh:.1f} dB (peak-25)")

    # contiguous speech regions
    regions = []
    start = None
    for i, s in enumerate(speech):
        if s and start is None:
            start = times[i]
        elif not s and start is not None:
            regions.append((start, times[i]))
            start = None
    if start is not None:
        regions.append((start, times[-1]))
    regions = [(a, b) for a, b in regions if (b - a) >= 0.30]
    print(f"speech regions ({len(regions)}):")
    for a, b in regions:
        print(f"  {a:6.2f} - {b:6.2f} s   ({b - a:4.2f} s)")

    # 0.5 s energy timeline
    step = 0.5
    print("energy timeline (0.5 s bins):")
    for t0 in np.arange(0, dur, step):
        m = (times >= t0) & (times < t0 + step)
        if not m.any():
            continue
        seg_db = db[m].mean()
        bar = "#" * int(max(0, seg_db - (peak_db - 45)))
        print(f"  {t0:5.1f}s  {seg_db:7.1f} dB  {bar}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(times, db, lw=0.7)
    axes[0].axhline(thresh, color="r", ls="--", lw=0.8, label=f"thr {thresh:.0f} dB")
    for a, b in regions:
        axes[0].axvspan(a, b, color="green", alpha=0.12)
    axes[0].set_title("Short-time RMS (dB)")
    axes[0].set_xlabel("time (s)")
    axes[0].legend(loc="lower right")

    axes[1].specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
    axes[1].set_title("Spectrogram")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("freq (Hz)")

    plt.tight_layout()
    out_png = wav_path.rsplit(".", 1)[0] + "_analysis.png"
    plt.savefig(out_png, dpi=110)
    plt.close()
    print(f"saved plot: {out_png}")


if __name__ == "__main__":
    main(*sys.argv[1:])
