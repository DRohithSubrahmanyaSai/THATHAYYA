import numpy as np
import torch


class SileroVad:
    name = "silero"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self.model.eval()

    def probability(self, chunk: np.ndarray, sample_rate: int) -> float:
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        if sample_rate != 16000:
            import librosa

            chunk = librosa.resample(chunk, orig_sr=sample_rate, target_sr=16000)
        window = 512
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        probs = []
        for i in range(0, len(chunk), window):
            w = chunk[i : i + window]
            if len(w) < window:
                w = np.pad(w, (0, window - len(w)))
            with torch.no_grad():
                probs.append(float(self.model(torch.from_numpy(w), 16000)))
        return max(probs) if probs else 0.0

    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> bool:
        return self.probability(chunk, sample_rate) > self.threshold


class EnergyVad:
    name = "energy"

    def __init__(self, threshold_db: float = -40.0):
        self.threshold_db = threshold_db

    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> bool:
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        rms = float(np.sqrt(np.mean(chunk**2)) + 1e-10)
        return 20 * np.log10(rms) > self.threshold_db
