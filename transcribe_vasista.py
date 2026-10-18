import os
import sys

import torch  # IMPORTANT: import torch first to avoid Windows c10.dll load error

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(r"D:\Thathayya_clone\.env")
os.environ.setdefault("HF_HOME", r"D:\Thathayya_clone\.cache\hf")

import soundfile as sf
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "vasista22/whisper-telugu-large-v2"
AUDIO = r"D:\Thathayya_clone\data\recordings\Thathayya.MP3"
OUT = r"D:\Thathayya_clone\data\ref_transcript_vasista.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading", MODEL_ID, "on", device)
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
model.eval()
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = None
model.generation_config.begin_suppress_tokens = None

data, sr = sf.read(AUDIO, dtype="float32")
if data.ndim > 1:
    data = data.mean(axis=1)
if sr != 16000:
    data = torchaudio.functional.resample(torch.from_numpy(data), sr, 16000).numpy()

feats = processor.feature_extractor(data, sampling_rate=16000, return_tensors="pt").input_features
feats = feats.to(device, dtype=dtype)

forced = processor.tokenizer.get_decoder_prompt_ids(language="telugu", task="transcribe")
print("Transcribing (Telugu)...")
with torch.no_grad():
    ids = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=440)
text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

print("\n--- VASISTA WHISPER TELUGU ---")
print(text)
print("--- END ---\n")

with open(OUT, "w", encoding="utf-8") as f:
    f.write(text + "\n")
print("Saved to", OUT)
