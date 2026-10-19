import asyncio
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from . import config
from .audio_input import MicSource, rms_int16
from .brain import Brain, build_system_prompt
from .learn import LiveLearner
from .memory import CallLog
from .player import StreamPlayer
from .sentences import SentenceStream
from .stt_live import DeepgramLiveSTT
from .tts_live import CartesiaTTSStream

_SENTINEL = object()


def _next_or(it, sentinel):
    try:
        return next(it)
    except StopIteration:
        return sentinel


class RealtimeAgent:
    def __init__(self, model: str | None = None, mute: bool = False):
        self.mute = mute
        self.brain = Brain(model=model)
        self.player = StreamPlayer()
        self.stt = DeepgramLiveSTT()
        self.tts = CartesiaTTSStream()
        self.mic = MicSource()
        self.log = CallLog()
        self.learner = LiveLearner(self.brain.memory, self.brain.client, model=config.GROQ_LEARN_MODEL)
        self.state = "listening"
        self._final_buf: list[str] = []
        self._conf: list[float] = []
        self._barge = asyncio.Event()
        self._ctx: str | None = None
        self._ignore_until = 0.0
        self._mic_task = None
        self._evt_task = None

    async def start(self) -> None:
        await self.stt.connect()
        await self.tts.connect()
        self.player.start()
        self.mic.start()
        self._mic_task = asyncio.create_task(self._mic_loop())
        self._evt_task = asyncio.create_task(self._event_loop())

    async def _mic_loop(self) -> None:
        need = max(1, int(config.BARGE_IN_MIN_S * 1000 / 32))
        loud = 0
        while True:
            block = await self.mic.read_async()
            if not block:
                continue
            if config.BARGE_IN and self.state == "speaking":
                if rms_int16(block) > config.BARGE_IN_RMS:
                    loud += 1
                    if loud >= need:
                        self._barge_in()
                        loud = 0
                else:
                    loud = 0
            else:
                loud = 0
            await self.stt.send_audio(block)

    def _barge_in(self) -> None:
        if self.state != "speaking":
            return
        self._barge.set()
        self.player.stop()
        if self._ctx:
            asyncio.ensure_future(self.tts.cancel(self._ctx))
        print("\n[barge-in]")

    async def _event_loop(self) -> None:
        while True:
            ev = await self.stt.events.get()
            kind = ev.get("type")
            if kind in ("interim", "final"):
                if self.state != "listening" or time.time() < self._ignore_until:
                    continue
                if kind == "interim":
                    continue
                self._final_buf.append(ev["text"])
                if ev.get("confidence") is not None:
                    self._conf.append(ev["confidence"])
                if ev.get("speech_final"):
                    await self._flush_turn()
            elif kind == "utterance_end":
                if self.state == "listening" and time.time() >= self._ignore_until:
                    await self._flush_turn()
            elif kind == "error":
                print(f"[stt error] {ev.get('error')}")
            elif kind == "closed":
                return

    async def _flush_turn(self) -> None:
        text = " ".join(self._final_buf).strip()
        conf = min(self._conf) if self._conf else None
        self._final_buf = []
        self._conf = []
        if not text:
            return
        self.log.write("user", text, confidence=conf)
        shown = f"{conf:.2f}" if conf is not None else "n/a"
        print(f"\nyou       : {text}   (conf={shown})")
        if conf is not None and conf < config.STT_CONFIDENCE_MIN:
            await self._speak_line(config.LOW_CONFIDENCE_REPLY)
            return
        await self._respond(text, conf)

    async def _speak_line(self, text: str) -> None:
        self.state = "speaking"
        self._barge.clear()
        self._ctx = uuid.uuid4().hex
        try:
            async for audio, sr in self.tts.speak(text, self._ctx):
                if self._barge.is_set():
                    break
                if not self.mute:
                    self.player.submit(audio, sr)
            if not self.mute and not self._barge.is_set():
                self.player.drain()
        except Exception as e:  # noqa: BLE001
            print(f"[tts error] {e}")
        finally:
            self._ctx = None
            self.state = "listening"
            self._ignore_until = time.time() + 0.3

    async def _respond(self, text: str, conf: float | None = None) -> None:
        # Note any durable fact into pending (you review; auto-kept at call end).
        if self.learner.maybe_learn(text, conf):
            for fact in self.brain.memory.pending():
                print(f"[pending] {fact}")
            self.brain.system = build_system_prompt(self.brain.persona, learned=self.brain.memory.as_prompt())

        self.state = "speaking"
        self._barge.clear()
        self._ignore_until = 0.0
        self._ctx = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        splitter = SentenceStream()
        gen = self.brain.stream(text)
        parts: list[str] = []
        t0 = time.time()
        first_audio = [None]

        async def sender():
            try:
                while True:
                    delta = await loop.run_in_executor(None, _next_or, gen, _SENTINEL)
                    if delta is _SENTINEL:
                        break
                    if self._barge.is_set():
                        return
                    parts.append(delta)
                    for sentence in splitter.push(delta):
                        line = LiveLearner.sanitize(sentence)
                        if line:
                            await self.tts.send_text(self._ctx, line, continue_=True)
                for sentence in splitter.flush():
                    if self._barge.is_set():
                        return
                    line = LiveLearner.sanitize(sentence)
                    if line:
                        await self.tts.send_text(self._ctx, line, continue_=True)
                await self.tts.send_text(self._ctx, "", continue_=False)
            except Exception as e:  # noqa: BLE001
                print(f"[llm/tts send error] {e}")

        send_task = asyncio.create_task(sender())
        try:
            async for audio, sr in self.tts.chunks(self._ctx):
                if self._barge.is_set():
                    break
                if first_audio[0] is None:
                    first_audio[0] = time.time() - t0
                if not self.mute:
                    self.player.submit(audio, sr)
        except Exception as e:  # noqa: BLE001
            print(f"[tts error] {e}")
        finally:
            if not self.mute and not self._barge.is_set():
                self.player.drain()
            send_task.cancel()
            try:
                gen.close()
            except Exception:
                pass

        reply = "".join(parts).strip()
        total = round(time.time() - t0, 2)
        fa = f"{first_audio[0]:.2f}s" if first_audio[0] is not None else "n/a"
        self.log.write("assistant", reply, first_audio=fa, total=total)
        print(f"Thathayya : {reply}")
        print(f"[latency] first_audio={fa}  total={total}s")
        self._ctx = None
        self.state = "listening"
        self._ignore_until = time.time() + 0.3

    async def run(self) -> None:
        await self.start()
        print(f"learned memories: {self.brain.memory.count()} | log: {self.log.path.name}")
        print("Ready. Speak into your mic. Ctrl+C to quit.\n")
        try:
            await asyncio.gather(self._mic_task, self._evt_task)
        except asyncio.CancelledError:
            pass
        finally:
            await self.close()

    async def close(self) -> None:
        self.mic.stop()
        await self.stt.close()
        await self.tts.close()
        self.player.close()
        # Auto-keep: promote all noted (pending) facts into long-term memory.
        kept = self.brain.memory.promote_all()
        if kept:
            print(f"\n[auto-kept {kept} noted fact(s) into learned memories")


def main(model: str | None = None, mute: bool = False) -> None:
    agent = RealtimeAgent(model=model, mute=mute)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nbye")
