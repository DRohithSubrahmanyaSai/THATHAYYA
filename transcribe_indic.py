import os
import sys

import torch  # IMPORTANT: import torch first to avoid Windows c10.dll load error

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(r"D:\Thathayya_clone\.env")
os.environ.setdefault("HF_HOME", r"D:\Thathayya_clone\.cache\hf")

from huggingface_hub import snapshot_download

MODEL_ID = "bodhan-ai/indic-transcribe-core"
AUDIO = sys.argv[1] if len(sys.argv) > 1 else r"D:\Thathayya_clone\data\recordings\Thathayya.MP3"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"D:\Thathayya_clone\data\ref_transcript_indic.txt"

print("Downloading / locating model:", MODEL_ID)
model_dir = snapshot_download(MODEL_ID)
print("model_dir:", model_dir)

sys.path.insert(0, model_dir)

# --- compatibility shim: on transformers 4.x the custom code calls
# from_pretrained(dtype=...) which is a v5 API; transformers 5.x needs no shim.
import transformers as _tf

if _tf.__version__.split(".")[0] == "4":
    import modeling_indic_canary as _mic

    _orig_fp = _mic.IndicCanaryForConditionalGeneration.from_pretrained

    @classmethod
    def _patched_fp(cls, *args, **kwargs):
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return _orig_fp.__func__(cls, *args, **kwargs)

    _mic.IndicCanaryForConditionalGeneration.from_pretrained = _patched_fp

from indic_transcribe import IndicTranscribe

print("Loading IndicTranscribe ...")
asr = IndicTranscribe.from_pretrained(model_dir)

print("Transcribing (lang=te):", AUDIO)
text, lid = asr(AUDIO, lang="te", return_lid=True)

print("\n--- INDIC-TRANSCRIBE (Telugu) ---")
print(text)
print("--- END ---")
print("LID:", lid, "\n")

with open(OUT, "w", encoding="utf-8") as f:
    f.write(text + "\n")
print("Saved to", OUT)
