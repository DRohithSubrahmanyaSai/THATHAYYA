import io

import numpy as np
import requests
import soundfile as sf

from . import config


class FasterWhisperSTT:
    name = "faster-whisper"

    def __init__(self, model_size: str = "large-v3", device: str = "cpu", language: str = "te"):
        self.language = language
        self.device = device
        compute = "float16" if device == "cuda" else "int8"
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device=self.device, compute_type=compute)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        segments, _ = self.model.transcribe(audio, language=self.language, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


class DeepgramSTT:
    name = "deepgram"

    def __init__(self, language: str = "te", model: str = "nova-3"):
        if not config.DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")
        self.language = language
        self.model = model
        self.url = f"https://api.deepgram.com/v1/listen?model={model}&language={language}&smart_format=true"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        pcm = np.clip(audio, -1, 1)
        pcm16 = (pcm * 32767).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, pcm16, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        r = requests.post(
            self.url,
            headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
            data=buf.read(),
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Deepgram {r.status_code}: {r.text}")
        return (r.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("paragraphs", {}).get("transcript") or r.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")).strip()


_ENGINES = {"faster-whisper": FasterWhisperSTT, "deepgram": DeepgramSTT}


class STT:
    def __init__(self, engine: str | None = None, **kwargs):
        self.engine_name = engine or config.STT_ENGINE
        if self.engine_name not in _ENGINES:
            raise ValueError(f"Unknown STT engine '{self.engine_name}'. Choose from {list(_ENGINES)}")
        self._engine = _ENGINES[self.engine_name](**kwargs)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return self._engine.transcribe(audio, sample_rate)
