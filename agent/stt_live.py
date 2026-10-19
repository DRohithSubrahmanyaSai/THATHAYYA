import asyncio
import json
from urllib.parse import urlencode

import websockets

from . import config
from .keyterms import get_keyterms


class DeepgramLiveSTT:
    """Deepgram streaming (WebSocket) transcription client.

    Push linear16 PCM via send_audio(); consume transcript events from the
    `events` asyncio.Queue. Event kinds:
      - {"type": "interim", "text": str}
      - {"type": "final", "text": str, "confidence": float, "speech_final": bool}
      - {"type": "utterance_end"}
      - {"type": "speech_started"}
      - {"type": "error", "error": str}
      - {"type": "closed"}
    """

    def __init__(self, sample_rate: int = 16000, model: str | None = None, language: str | None = None):
        if not config.DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")
        self.sample_rate = sample_rate
        self.model = model or config.DEEPGRAM_MODEL
        self.language = language or config.STT_LANGUAGE
        self.events: "asyncio.Queue[dict]" = asyncio.Queue()
        self._ws = None
        self._recv_task = None
        self._keepalive_task = None

    def _url(self) -> str:
        params = [
            ("model", self.model),
            ("language", self.language),
            ("encoding", "linear16"),
            ("sample_rate", str(self.sample_rate)),
            ("channels", "1"),
            ("smart_format", "true"),
            ("punctuate", "true"),
            ("interim_results", "true"),
            ("vad_events", "true"),
            ("endpointing", str(config.DG_ENDPOINTING_MS)),
            ("utterance_end_ms", str(config.DG_UTTERANCE_END_MS)),
        ]
        for term in get_keyterms():
            params.append(("keyterm", term))
        return "wss://api.deepgram.com/v1/listen?" + urlencode(params)

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self._url(),
            additional_headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"},
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(config.DG_KEEPALIVE_S)
            await self.keepalive()

    async def _recv_loop(self) -> None:
        try:
            async for message in self._ws:
                data = json.loads(message)
                kind = data.get("type")
                if kind == "Results":
                    alt = data.get("channel", {}).get("alternatives", [{}])[0]
                    text = (alt.get("transcript") or "").strip()
                    if not text:
                        continue
                    speech_final = bool(data.get("speech_final"))
                    is_final = bool(data.get("is_final")) or speech_final
                    await self.events.put(
                        {
                            "type": "final" if is_final else "interim",
                            "text": text,
                            "confidence": alt.get("confidence"),
                            "speech_final": speech_final,
                        }
                    )
                elif kind == "UtteranceEnd":
                    await self.events.put({"type": "utterance_end"})
                elif kind == "SpeechStarted":
                    await self.events.put({"type": "speech_started"})
        except Exception as e:  # noqa: BLE001
            await self.events.put({"type": "error", "error": str(e)})
        finally:
            await self.events.put({"type": "closed"})

    async def send_audio(self, pcm16: bytes) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(pcm16)
            except Exception:
                pass

    async def keepalive(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
            except Exception:
                pass

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._recv_task is not None:
            self._recv_task.cancel()
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
