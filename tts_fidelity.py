"""IndicF5 fidelity-focused synthesis.

Bypasses INF5Model.forward() (which hardcodes cfg=2.0, nfe=32, no seed and
internally re-clips the reference) and calls f5_tts.infer_process directly with
a pre-cleaned, pre-aligned reference so we can tune every generation knob and
seed the sampler for reproducibility.

Import order: torch FIRST (Windows c10.dll load order).
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse
import time

import torch  # noqa: F401  (must be imported first)

from dotenv import load_dotenv

load_dotenv(r"D:\Thathayya_clone\.env")
os.environ.setdefault("HF_HOME", r"D:\Thathayya_clone\.cache\hf")

import numpy as np
import soundfile as sf
import torchaudio
from pydub import AudioSegment, silence

# --- torchaudio 2.11 has no native decoders -> shim .load to soundfile ---
def _sf_load(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _sf_load

from f5_tts.infer.utils_infer import (  # noqa: E402
    infer_process,
    load_vocoder,
    preprocess_ref_audio_text,
)

MODEL_ID = "ai4bharat/IndicF5"
DEFAULT_REF = r"D:\Thathayya_clone\data\refs\reference_thathayya.wav"
DEFAULT_REFTEXT = r"D:\Thathayya_clone\data\refs\reference_thathayya.txt"

_model_cache = {}


def load_indicf5(model_id: str = MODEL_ID):
    """Load IndicF5 once; return dict with ema_model + vocoder on the right device."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    from transformers import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {model_id} on {device} ...")
    t0 = time.time()
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    ema_model = model.ema_model
    vocoder = model.vocoder
    print(f"[load] done in {time.time() - t0:.1f}s")
    bundle = {"ema_model": ema_model, "vocoder": vocoder, "device": device, "model": model}
    _model_cache[model_id] = bundle
    return bundle


def _postprocess(audio_float: np.ndarray, sr: int, remove_sil: bool = True, target_dbfs: float = -20.0):
    buf = _to_pydub(audio_float, sr)
    if remove_sil:
        segs = silence.split_on_silence(buf, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10)
        buf = sum(segs, AudioSegment.silent(duration=0))
    change = target_dbfs - buf.dBFS
    buf = buf.apply_gain(change)
    arr = np.array(buf.get_array_of_samples()).astype(np.float32)
    arr = arr / (32768.0 if buf.sample_width == 2 else float(2 ** (8 * buf.sample_width - 1)))
    return arr, buf.frame_rate


def _to_pydub(audio_float: np.ndarray, sr: int) -> AudioSegment:
    a = np.clip(audio_float, -1.0, 1.0)
    pcm = (a * 32767.0).astype(np.int16)
    import io

    bio = io.BytesIO()
    sf.write(bio, pcm, sr, format="WAV", subtype="PCM_16")
    bio.seek(0)
    return AudioSegment.from_file(bio, format="wav")


def synth(
    text: str,
    ref: str = DEFAULT_REF,
    reftext: str = DEFAULT_REFTEXT,
    cfg_strength: float = 2.0,
    nfe_step: int = 32,
    sway_sampling_coef: float = -1.0,
    speed: float = 1.0,
    seed: int | None = 0,
    remove_sil: bool = True,
    model_id: str = MODEL_ID,
    preprocess: bool = False,
):
    """Generate speech. Returns (float32 mono numpy, sample_rate).

    preprocess=True replicates the original INF5Model.forward() path: the
    reference file is passed through f5_tts preprocess_ref_audio_text (silence
    trim + clip to <=15s + re-encode). This is the pipeline the user preferred.
    """
    bundle = load_indicf5(model_id)
    with open(reftext, "r", encoding="utf-8") as f:
        rt = f.read().strip()

    if preprocess:
        ref, rt = preprocess_ref_audio_text(ref, rt, show_info=lambda *a, **k: None)

    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    with torch.inference_mode():
        audio, sr, _ = infer_process(
            ref,
            rt,
            text,
            bundle["ema_model"],
            bundle["vocoder"],
            mel_spec_type="vocos",
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            device=bundle["device"],
        )

    audio = np.asarray(audio, dtype=np.float32)
    if remove_sil:
        audio, sr = _postprocess(audio, sr, remove_sil=True)
    return audio, sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default=DEFAULT_REF)
    p.add_argument("--reftext", default=DEFAULT_REFTEXT)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--sway", type=float, default=-1.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-post", action="store_true")
    p.add_argument("--preprocess", action="store_true", help="replicate original forward() reference preprocessing")
    args = p.parse_args()

    print(f"[synth] cfg={args.cfg} nfe={args.nfe} sway={args.sway} speed={args.speed} seed={args.seed} preprocess={args.preprocess}")
    t0 = time.time()
    audio, sr = synth(
        args.text,
        ref=args.ref,
        reftext=args.reftext,
        cfg_strength=args.cfg,
        nfe_step=args.nfe,
        sway_sampling_coef=args.sway,
        speed=args.speed,
        seed=args.seed,
        remove_sil=not args.no_post,
        preprocess=args.preprocess,
    )
    sf.write(args.out, audio, sr)
    print(f"[synth] wrote {args.out}  {len(audio)/sr:.2f}s @ {sr}Hz  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
