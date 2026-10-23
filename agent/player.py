import asyncio
import queue
import threading
import time
from math import gcd

import numpy as np
import sounddevice as sd

from . import config

try:
    from scipy.signal import resample_poly

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


class StreamPlayer:
    """Persistent, interruptible output stream.

    Feed mono float32 audio via submit(); it plays gaplessly. stop() clears
    everything instantly (for barge-in).
    """

    def __init__(self, sample_rate: int | None = None, block_ms: int = 20):
        self.sr = int(sample_rate or config.PLAYER_SR)
        self.block_ms = block_ms
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._leftover = np.zeros(0, dtype="float32")
        self._stream = None

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=0,
            callback=self._callback,
        )
        self._stream.start()

    def _ensure(self) -> None:
        if self._stream is None:
            self.start()

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = np.asarray(audio, dtype="float32").flatten()
        if sr == self.sr or audio.size == 0:
            return audio
        if _HAVE_SCIPY:
            g = gcd(int(sr), self.sr)
            return resample_poly(audio, self.sr // g, sr // g).astype("float32")
        n = int(round(audio.size * self.sr / sr))
        return np.interp(
            np.linspace(0, audio.size, n, endpoint=False),
            np.arange(audio.size),
            audio,
        ).astype("float32")

    def submit(self, audio: np.ndarray, sr: int) -> None:
        self._ensure()
        self._q.put(self._resample(audio, sr))

    def _callback(self, outdata, frames, time_info, status):  # noqa: ARG002
        out = outdata[:, 0]
        filled = 0
        while filled < frames:
            if self._leftover.size == 0:
                try:
                    self._leftover = self._q.get_nowait()
                except queue.Empty:
                    break
                if self._leftover.ndim == 0:
                    break
            take = min(frames - filled, self._leftover.size)
            out[filled : filled + take] = self._leftover[:take]
            self._leftover = self._leftover[take:]
            filled += take
        if filled < frames:
            out[filled:] = 0.0

    def stop(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._leftover = np.zeros(0, dtype="float32")

    def is_playing(self) -> bool:
        return (not self._q.empty()) or self._leftover.size > 0

    def drain(self, grace_s: float = 0.12) -> None:
        while not self._q.empty() or self._leftover.size > 0:
            time.sleep(0.01)
        time.sleep(grace_s)

    async def drain_async(self, grace_s: float = 0.12) -> None:
        """Like drain() but never blocks the asyncio loop (keeps mic/STT live)."""
        while not self._q.empty() or self._leftover.size > 0:
            await asyncio.sleep(0.01)
        await asyncio.sleep(grace_s)

    def play(self, audio: np.ndarray, sr: int) -> None:
        self.submit(audio, sr)
        self.drain()

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stream = None
