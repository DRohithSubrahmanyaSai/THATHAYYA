import asyncio
import queue

import numpy as np
import sounddevice as sd


class MicSource:
    """16 kHz mono int16 microphone stream with a thread-safe block queue."""

    def __init__(self, sample_rate: int = 16000, block_ms: int = 32):
        self.sr = sample_rate
        self.block = int(sample_rate * block_ms / 1000)
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=400)
        self._stream = None

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        try:
            self._q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def start(self) -> None:
        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype="int16",
                blocksize=self.block,
                callback=self._callback,
            )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stream = None

    def read(self, timeout: float = 0.5) -> bytes:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return b""

    async def read_async(self, executor=None) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.read)


def rms_int16(block: bytes) -> float:
    if not block:
        return 0.0
    a = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a**2)))
