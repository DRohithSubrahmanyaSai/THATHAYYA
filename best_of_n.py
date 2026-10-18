"""Best-of-N synthesis: generate many takes over a small (cfg, nfe, seed) grid,
rank them by speaker-embedding cosine similarity to the reference voice, and
keep the best.

Import order: torch FIRST.
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse
import itertools
import json
import time

import torch  # noqa: F401  (must be first)

from dotenv import load_dotenv

load_dotenv(r"D:\Thathayya_clone\.env")
os.environ.setdefault("HF_HOME", r"D:\Thathayya_clone\.cache\hf")

import numpy as np
import soundfile as sf
import torchaudio

import tts_fidelity as tf

SV_MODEL = "microsoft/wavlm-base-plus-sv"
_sv = {}


def load_sv():
    if "bundle" not in _sv:
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        model = AutoModelForAudioXVector.from_pretrained(SV_MODEL).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _sv["bundle"] = (fe, model, device)
    return _sv["bundle"]


def embed(wav: np.ndarray, sr: int) -> np.ndarray:
    fe, model, device = load_sv()
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.from_numpy(wav).float(), sr, 16000).numpy()
    inp = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        out = model(**inp)
    e = out.embeddings.squeeze(0).cpu().numpy()
    e = e / (np.linalg.norm(e) + 1e-9)
    return e


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--ref", default=tf.DEFAULT_REF)
    p.add_argument("--reftext", default=tf.DEFAULT_REFTEXT)
    p.add_argument("--cfg", default="1.5,2.0,2.5")
    p.add_argument("--nfe", default="32,48")
    p.add_argument("--seed", default="0,1")
    p.add_argument("--sway", type=float, default=-1.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--save-all", action="store_true")
    p.add_argument("--preprocess", action="store_true", help="replicate original forward() reference preprocessing")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ref_audio, ref_sr = sf.read(args.ref, dtype="float32")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    ref_emb = embed(ref_audio, ref_sr)
    print(f"[ref] {args.ref}  {len(ref_audio)/ref_sr:.2f}s  emb[done]")

    cfgs = [float(x) for x in args.cfg.split(",")]
    nfes = [int(x) for x in args.nfe.split(",")]
    seeds = [int(x) for x in args.seed.split(",")]
    combos = list(itertools.product(cfgs, nfes, seeds))
    print(f"[grid] {len(combos)} takes: cfg={cfgs} nfe={nfes} seeds={seeds} sway={args.sway} speed={args.speed}")

    results = []
    t_all = time.time()
    for i, (cfg, nfe, seed) in enumerate(combos, 1):
        t0 = time.time()
        audio, sr = tf.synth(
            args.text,
            ref=args.ref,
            reftext=args.reftext,
            cfg_strength=cfg,
            nfe_step=nfe,
            sway_sampling_coef=args.sway,
            speed=args.speed,
            seed=seed,
            preprocess=args.preprocess,
        )
        emb = embed(audio, sr)
        sim = cos(ref_emb, emb)
        name = f"cfg{cfg}_nfe{nfe}_seed{seed}.wav"
        sf.write(os.path.join(args.outdir, name), audio, sr)
        results.append({"name": name, "cfg": cfg, "nfe": nfe, "seed": seed, "sim": sim, "dur": len(audio) / sr, "sr": sr})
        print(f"  [{i:2d}/{len(combos)}] {name:32s} sim={sim:.4f}  dur={len(audio)/sr:5.2f}s  ({time.time()-t0:.0f}s)")

    results.sort(key=lambda r: r["sim"], reverse=True)
    best = results[0]
    sf.write(os.path.join(args.outdir, "BEST.wav"), sf.read(os.path.join(args.outdir, best["name"]))[0], best["sr"])

    if not args.save_all:
        for r in results:
            if r["name"] != best["name"]:
                try:
                    os.remove(os.path.join(args.outdir, r["name"]))
                except OSError:
                    pass

    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"text": args.text, "ref": args.ref, "reftext_file": args.reftext, "wc": window_sig(args.ref), "ranking": results}, f, ensure_ascii=False, indent=2)

    print(f"\n[best] {best['name']}  sim={best['sim']:.4f}  dur={best['dur']:.2f}s")
    print(f"[total] {time.time()-t_all:.0f}s  -> {args.outdir}")
    print("[top5]")
    for r in results[:5]:
        print(f"   {r['name']:32s} sim={r['sim']:.4f}")


def window_sig(path: str) -> dict:
    try:
        import hashlib

        return {"path": path, "md5": hashlib.md5(open(path, "rb").read()).hexdigest()}
    except OSError:
        return {"path": path}


if __name__ == "__main__":
    main()
