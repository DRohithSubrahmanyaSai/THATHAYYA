import numpy as np
import sounddevice as sd

DEFAULT_SR = 16000


def record(duration: float, sr: int = DEFAULT_SR) -> np.ndarray:
    return sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32").flatten()


def play(audio: np.ndarray, sr: int) -> None:
    sd.play(np.asarray(audio, dtype=np.float32), sr)
    sd.wait()


def listen_until_silence(
    vad,
    sr: int = DEFAULT_SR,
    chunk_ms: int = 32,
    max_silence_ms: int = 900,
    max_record_s: float = 30.0,
    prefix_s: float = 0.3,
):
    chunk_samples = int(sr * chunk_ms / 1000)
    max_silence_chunks = max(1, int(max_silence_ms / chunk_ms))
    max_chunks = int(max_record_s * 1000 / chunk_ms)
    prefix_chunks = int(prefix_s * 1000 / chunk_ms)

    frames = []
    speech_started = False
    silence_chunks = 0
    total = 0

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32") as stream:
        while total < max_chunks:
            data, _ = stream.read(chunk_samples)
            chunk = data.flatten()
            frames.append(chunk)
            total += 1
            if vad.is_speech(chunk, sr):
                if not speech_started and len(frames) > prefix_chunks:
                    frames = frames[-prefix_chunks:]
                speech_started = True
                silence_chunks = 0
            elif speech_started:
                silence_chunks += 1
                if silence_chunks >= max_silence_chunks:
                    break

    if not frames:
        return np.zeros(0, dtype=np.float32), sr
    return np.concatenate(frames).astype(np.float32), sr
