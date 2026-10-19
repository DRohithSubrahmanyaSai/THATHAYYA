import asyncio
import base64
import json

import numpy as np
import websockets

from . import config


class CartesiaTTSStream:
    """Cartesia streaming TTS over WebSocket.

    Send text incrementally with send_text(..., continue_=True) and a final
    send_text(..., continue_=False); iterate chunks() for raw PCM float32 audio.
    """

    def __init__(self):
        if not config.CARTESIA_API_KEY:
            raise RuntimeError("CARTESIA_API_KEY is not set in .env")
        self.voice_id = config.CARTESIA_VOICE_ID or "bdaae173-3796-4f4d-9ba1-cc94c7de90c8"
        self.model_id = config.CARTESIA_MODEL
        self.sample_rate = config.CARTESIA_SR
        self._ws = None
        self._recv_task = None
        self._queue: "asyncio.Queue[dict]" = asyncio.Queue()

    def _url(self) -> str:
        return f"wss://api.cartesia.ai/tts/websocket?cartesia_version={config.CARTESIA_VERSION}"

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self._url(),
            additional_headers={
                "X-API-Key": config.CARTESIA_API_KEY,
                "Cartesia-Version": config.CARTESIA_VERSION,
            },
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            async for message in self._ws:
                await self._queue.put(json.loads(message))
        except Exception as e:  # noqa: BLE001
            await self._queue.put({"type": "error", "error": str(e)})
        finally:
            await self._queue.put({"type": "closed"})

    async def send_text(self, context_id: str, text: str, continue_: bool = True) -> None:
        payload = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": self.voice_id,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "context_id": context_id,
            "continue": continue_,
            "add_timestamps": False,
            "speed": config.CARTESIA_SPEED,
            "scale": config.CARTESIA_SCALE,
            "temperature": config.CARTESIA_TEMPERATURE,
        }
        await self._ws.send(json.dumps(payload))

    async def cancel(self, context_id: str) -> None:
        try:
            await self._ws.send(json.dumps({"context_id": context_id, "cancel": True}))
        except Exception:
            pass

    async def chunks(self, context_id: str, timeout: float = 30.0):
        while True:
            try:
                data = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return
            kind = data.get("type")
            if kind == "chunk":
                if data.get("context_id") != context_id:
                    continue
                raw = base64.b64decode(data.get("data") or "")
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                yield audio, self.sample_rate
            elif kind == "done":
                if data.get("context_id") == context_id:
                    return
            elif kind == "error":
                msg = str(data.get("message") or "")
                if data.get("done") or "cancelled" in msg.lower() or "does not exist" in msg.lower():
                    return
                raise RuntimeError(f"Cartesia WS error: {data}")
            elif kind == "closed":
                return

    async def speak(self, text: str, context_id: str):
        """Convenience: send one complete line and yield audio chunks."""
        await self.send_text(context_id, text, continue_=False)
        async for item in self.chunks(context_id):
            yield item

    async def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
