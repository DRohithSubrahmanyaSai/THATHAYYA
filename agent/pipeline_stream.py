import asyncio
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from . import config, intent, pronunciation
from .audio_input import MicSource, rms_int16
from .brain import Brain
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
        self.pron = pronunciation.get_store()
        self.state = "listening"
        self._final_buf: list[str] = []
        self._conf: list[float] = []
        self._barge = asyncio.Event()
        self._ctx: str | None = None
        self._ignore_until = 0.0
        self._mic_task = None
        self._evt_task = None
        self._audio = bytearray()
        self._max_audio = int(16000 * 2 * config.PRON_REF_S)
        self._bg_tasks: set = set()

    async def start(self) -> None:
        self.brain.memory.reset_live()
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
            self._audio.extend(block)
            if len(self._audio) > self._max_audio:
                del self._audio[: -self._max_audio]
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
        kind = intent.classify(text, conf)
        self.log.write("user", text, confidence=conf, intent=kind)
        shown = f"{conf:.2f}" if conf is not None else "n/a"
        print(f"\nyou       : {text}   (conf={shown}, intent={kind})")

        if kind == "unclear":
            await self._speak_line(config.LOW_CONFIDENCE_REPLY, intent_kind=kind)
            return
        if kind == "unknown_person":
            await self._speak_line(config.UNKNOWN_PERSON_REPLY, intent_kind=kind)
            return
        if kind == "correction":
            self._schedule_correction(text)
            await self._speak_line(config.CORRECTION_ACK, intent_kind=kind)
            return
        if kind == "teach":
            fact = self.brain.remember_live(text)
            if fact:
                print(f"[live-learned] {fact}")
            await self._respond(text, conf, learn=False, intent_kind=kind)
            return
        await self._respond(text, conf, learn=True, intent_kind=kind)

    def _schedule_correction(self, text: str) -> None:
        audio = bytes(self._audio[-int(16000 * 2 * config.PRON_REF_S):])
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, self._learn_correction, text, audio)
        self._bg_tasks.add(fut)
        fut.add_done_callback(self._bg_tasks.discard)

    def _learn_correction(self, text: str, audio: bytes) -> None:
        try:
            pair = self.pron.extract(text, client=self.brain.client)
            if not pair:
                return
            wrong, right = pair
            ref = self.pron.save_ref(audio, 16000, wrong, right)
            if self.pron.add(wrong, right, ref=ref):
                self.brain.refresh_system()
                print(f"\n[pron] {wrong} -> {right}")
        except Exception as e:  # noqa: BLE001
            print(f"[pron error] {e}")

    def _learn_async(self, text: str, conf: float | None) -> None:
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, self._learn_sync, text, conf)
        self._bg_tasks.add(fut)
        fut.add_done_callback(self._bg_tasks.discard)

    def _learn_sync(self, text: str, conf: float | None) -> None:
        try:
            fact = self.learner.maybe_learn(text, conf)
            if fact:
                self.brain.refresh_system()
                print(f"\n[live-learned] {fact}")
        except Exception as e:  # noqa: BLE001
            print(f"[learn error] {e}")

    async def _speak_line(self, text: str, intent_kind: str = "fixed") -> None:
        self.state = "speaking"
        self._barge.clear()
        self._ctx = uuid.uuid4().hex
        print(f"Thathayya : {text}")
        t0 = time.time()
        first_audio = [None]
        try:
            async for audio, sr in self.tts.speak(text, self._ctx):
                if self._barge.is_set():
                    break
                if first_audio[0] is None:
                    first_audio[0] = time.time() - t0
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
        fa = f"{first_audio[0]:.2f}s" if first_audio[0] is not None else "n/a"
        self.log.write("assistant", text, first_audio=fa, intent=intent_kind)

    async def _respond(
        self,
        text: str,
        conf: float | None = None,
        learn: bool = True,
        intent_kind: str = "normal",
    ) -> None:
        if learn:
            self._learn_async(text, conf)

        self.state = "speaking"
        self._barge.clear()
        self._ignore_until = 0.0
        self._ctx = uuid.uuid4().hex
        ctx = self._ctx
        loop = asyncio.get_running_loop()
        splitter = SentenceStream()
        gen = self.brain.stream(text)
        parts: list[str] = []
        spoken: list[str] = []
        t0 = time.time()
        first_audio = [None]
        first_token = [None]
        sent = [0]
        max_sent = max(1, config.REPLY_MAX_SENTENCES)
        max_chars = max(20, config.REPLY_MAX_CHARS)

        async def emit(sentence: str) -> None:
            line = LiveLearner.sanitize(sentence)
            line = self.pron.apply(line).strip()
            if len(line) > max_chars:
                line = line[:max_chars].rsplit(" ", 1)[0] or line[:max_chars]
            line = line.strip()
            if not line:
                return
            spoken.append(line)
            sent[0] += 1
            await self.tts.send_text(ctx, line, continue_=True)

        async def sender() -> None:
            nonlocal gen
            try:
                first = True
                while True:
                    if first:
                        try:
                            delta = await asyncio.wait_for(
                                loop.run_in_executor(None, _next_or, gen, _SENTINEL),
                                timeout=config.LLM_FIRST_TOKEN_TIMEOUT_S,
                            )
                        except asyncio.TimeoutError:
                            print("[llm slow -> fallback]")
                            try:
                                gen.close()
                            except Exception:
                                pass
                            gen = self.brain.stream_fallback(text)
                            delta = await loop.run_in_executor(None, _next_or, gen, _SENTINEL)
                        first = False
                    else:
                        delta = await loop.run_in_executor(None, _next_or, gen, _SENTINEL)
                    if delta is _SENTINEL:
                        break
                    if self._barge.is_set():
                        return
                    if first_token[0] is None:
                        first_token[0] = time.time() - t0
                    parts.append(delta)
                    for sentence in splitter.push(delta):
                        if self._barge.is_set():
                            return
                        await emit(sentence)
                        if sent[0] >= max_sent:
                            await self.tts.send_text(ctx, "", continue_=False)
                            return
                for sentence in splitter.flush():
                    if self._barge.is_set():
                        return
                    await emit(sentence)
                    if sent[0] >= max_sent:
                        break
                await self.tts.send_text(ctx, "", continue_=False)
            except Exception as e:  # noqa: BLE001
                print(f"[llm/tts send error] {e}")
                try:
                    await self.tts.send_text(ctx, "", continue_=False)
                except Exception:
                    pass

        send_task = asyncio.create_task(sender())
        try:
            async for audio, sr in self.tts.chunks(ctx):
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

        reply = " ".join(spoken).strip()
        if not reply:
            reply = config.EMPTY_REPLY
            self.brain.remember_turn(text, reply)
            await self._speak_line(reply, intent_kind="empty")
            return
        self.brain.remember_turn(text, reply)
        total = round(time.time() - t0, 2)
        fa = f"{first_audio[0]:.2f}s" if first_audio[0] is not None else "n/a"
        ft = f"{first_token[0]:.2f}s" if first_token[0] is not None else "n/a"
        self.log.write(
            "assistant",
            reply,
            first_audio=fa,
            first_token=ft,
            total=total,
            intent=intent_kind,
            sentences=sent[0],
        )
        print(f"Thathayya : {reply}")
        print(f"[latency] first_token={ft} first_audio={fa}  total={total}s")
        self._ctx = None
        self.state = "listening"
        self._ignore_until = time.time() + 0.3

    async def run(self) -> None:
        await self.start()
        print(
            f"learned memories: {self.brain.memory.count()} | "
            f"pronunciation rules: {self.pron.count()} | log: {self.log.path.name}"
        )
        print("Ready. Speak into your mic. Ctrl+C to quit.\n")
        try:
            await asyncio.gather(self._mic_task, self._evt_task)
        except asyncio.CancelledError:
            pass
        finally:
            await self.close()

    async def close(self) -> None:
        for fut in list(self._bg_tasks):
            try:
                await fut
            except Exception:
                pass
        self.mic.stop()
        await self.stt.close()
        await self.tts.close()
        self.player.close()
        kept = self.brain.memory.promote_all()
        if kept:
            print(f"\n[auto-kept {kept} live/pending fact(s) into learned memories]")


def main(model: str | None = None, mute: bool = False) -> None:
    agent = RealtimeAgent(model=model, mute=mute)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nbye")
