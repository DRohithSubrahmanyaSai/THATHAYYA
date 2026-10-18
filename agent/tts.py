import io
import os

import requests
import torch

import soundfile as sf
import torchaudio

from . import config


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _soundfile_load


def _to_float32(audio) -> "np.ndarray":
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        audio = audio.astype("float32") / 32768.0
    return audio.astype("float32")


class LocalIndicF5:
    name = "indicf5_local"

    def __init__(self):
        self._model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = config.VOICE.get("indicf5_local", {})
        self.model_id = cfg.get("model", "ai4bharat/IndicF5")
        self.ref_audio = config.BASE / cfg.get("ref_audio", "data/recordings/Thathayya.MP3")
        self.ref_text = (config.BASE / cfg.get("ref_text_file", "data/ref_transcript.txt")).read_text(encoding="utf-8").strip()
        self.cfg = cfg.get("cfg", 2.0)
        self.nfe = cfg.get("nfe", 32)
        self.sample_rate = cfg.get("sample_rate", 24000) if "sample_rate" in cfg else 24000

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if config.HF_TOKEN:
            os.environ["HF_TOKEN"] = config.HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = config.HF_TOKEN
        os.environ.setdefault("HF_HOME", str(config.BASE / ".cache" / "hf"))
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        try:
            self._model.to(self.device)
        except Exception:
            pass

    def synthesize(self, text: str, speed: float = 1.0):
        self._ensure_loaded()
        self._model.config.speed = speed
        audio = self._model(text, ref_audio_path=str(self.ref_audio), ref_text=self.ref_text)
        return _to_float32(audio), 24000


class CartesiaTTS:
    name = "cartesia"

    def __init__(self):
        if not config.CARTESIA_API_KEY:
            raise RuntimeError("CARTESIA_API_KEY is not set in .env")
        self.voice_id = config.CARTESIA_VOICE_ID or "bdaae173-3796-4f4d-9ba1-cc94c7de90c8"
        self.model_id = config.CARTESIA_MODEL
        self.sample_rate = config.VOICE.get("cartesia", {}).get("sample_rate", 44100)
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": config.CARTESIA_API_KEY,
            "Cartesia-Version": "2026-08-14",
        }

    def synthesize(self, text: str, speed: float = 1.0):
        payload = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": self.voice_id,
            "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": self.sample_rate},
            "generation_config": {"speed": speed, "volume": 1},
        }
        r = requests.post("https://api.cartesia.ai/tts/bytes", json=payload, headers=self.headers, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Cartesia {r.status_code}: {r.text}")
        data, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.ndim > 1 else data
        return mono.astype("float32"), sr


_ENGINES = {"indicf5_local": LocalIndicF5, "cartesia": CartesiaTTS}


class TTS:
    def __init__(self, engine: str | None = None):
        self.engine_name = engine or config.TTS_DRAFT_ENGINE
        if self.engine_name not in _ENGINES:
            raise ValueError(f"Unknown TTS engine '{self.engine_name}'. Choose from {list(_ENGINES)}")
        self._engine = _ENGINES[self.engine_name]()

    @property
    def sample_rate(self):
        return self._engine.sample_rate

    def synthesize(self, text: str, speed: float = 1.0):
        return self._engine.synthesize(text, speed=speed)

    def synthesize_to_file(self, text: str, path, speed: float = 1.0):
        audio, sr = self.synthesize(text, speed=speed)
        sf.write(str(path), audio, sr)
        return path
