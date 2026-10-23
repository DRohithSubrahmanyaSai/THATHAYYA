"""Fast checks for the live-quality fixes (no network)."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import config, pronunciation
from agent.pipeline_stream import RealtimeAgent
from agent.sentences import SentenceStream, ends_sentence
from agent.tts_live import CartesiaTTSStream


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))


class FakeLog:
    def write(self, *args, **kwargs):
        pass


def check(name: str, cond: bool, detail: str = "") -> int:
    print(f"[{'OK ' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    return 0 if cond else 1


class FakeTTS:
    def __init__(self):
        self.sent = []
        self.done = asyncio.Event()

    async def send_text(self, ctx, text, continue_=True):
        self.sent.append(text)
        if not continue_:
            self.done.set()

    async def chunks(self, ctx):
        await asyncio.wait_for(self.done.wait(), timeout=2)
        if False:
            yield None, 24000

    async def speak(self, text, ctx):
        if False:
            yield None, 24000

    async def cancel(self, ctx):
        pass


class FakePlayer:
    async def drain_async(self, *args, **kwargs):
        pass


class FakePron:
    def apply(self, text):
        return text


def streaming_check() -> int:
    agent = RealtimeAgent.__new__(RealtimeAgent)
    agent.mute = True
    agent.state = "listening"
    agent._barge = asyncio.Event()
    agent._ignore_until = 0.0
    agent._ctx = None
    agent._turn_started = 0.0
    agent._final_buf = []
    agent._flush_task = None
    agent.log = FakeLog()
    agent.pron = FakePron()
    agent.tts = FakeTTS()
    agent.player = FakePlayer()
    agent._learn_async = lambda *a, **k: None
    agent._schedule_deferred_flush = lambda: None
    agent.brain = SimpleNamespace(
        stream=lambda text: iter(["అవును బాబూ, నేను అన్నం తిన్నాను."]),
        remember_turn=lambda *a: None,
    )
    asyncio.run(agent._respond("ఏం తిన్నావ్?", None, learn=False))
    lines = [t for t in agent.tts.sent if t]
    ok = lines == ["అవును బాబూ", "నేను అన్నం తిన్నాను"]
    return check("stream sends clause then one sentence", ok, str(lines))


def routing_check() -> int:
    fails = 0
    agent = RealtimeAgent.__new__(RealtimeAgent)
    agent.log = FakeLog()
    agent._conf = []
    agent.brain = SimpleNamespace(
        last_reply=lambda: "నేను అన్నం తిన్నాను.", remember_turn=lambda *a: None
    )
    spoke = []
    repairs = []

    async def speak(text, intent_kind="fixed"):
        spoke.append((text, intent_kind))

    async def respond(*args, **kwargs):
        raise AssertionError("complaint/repeat must not reach the LLM reply path")

    agent._speak_line = speak
    agent._respond = respond
    agent._schedule_repair = lambda text: repairs.append(text)
    agent._schedule_correction = lambda text: None

    agent._final_buf = ["గోరెండ్ బియ్యం అసలు ఉండదు కదండీ పిచ్చి పిచ్చిగా మాట్లాడుతున్నారు"]
    asyncio.run(agent._flush_turn())
    fails += check(
        "complaint uses repair line",
        spoke and spoke[-1] == (config.REPAIR_REPLY, "complaint"),
        str(spoke[-1] if spoke else None),
    )
    fails += check("complaint schedules negative learning", bool(repairs))

    spoke.clear()
    agent._final_buf = ["అదే ఏంటి?"]
    asyncio.run(agent._flush_turn())
    fails += check(
        "repeat speaks last reply",
        spoke and spoke[-1][1] == "repeat" and "అన్నం" in spoke[-1][0],
        str(spoke[-1] if spoke else None),
    )
    return fails


def main() -> int:
    fails = 0

    splitter = SentenceStream(first_clause_chars=22, first_clause_min=6)
    first = splitter.push("అవును బాబూ, నేను అన్నం")
    fails += check("early clause emitted at comma", first == ["అవును బాబూ,"], str(first))
    rest = splitter.push(" తిన్నాను.")
    fails += check("rest of sentence follows", rest == ["నేను అన్నం తిన్నాను."], str(rest))
    fails += check("clause is not a sentence", not ends_sentence("అవును బాబూ,"))
    fails += check("sentence ends", ends_sentence("తిన్నాను."))

    no_comma = SentenceStream(first_clause_chars=22, first_clause_min=6)
    fails += check(
        "no early split without comma",
        no_comma.push("నేను బాగున్నాను") == [],
        str(no_comma.push("")),
    )
    fails += check("flush emits remainder", no_comma.flush() == ["నేను బాగున్నాను"])

    text = "అ" * 200
    truncated = RealtimeAgent._truncate(text)
    fails += check("reply truncate", len(truncated) <= config.REPLY_MAX_CHARS, str(len(truncated)))

    tmp = Path(tempfile.mkdtemp(prefix="thathayya_pron_")) / "pron.json"
    store = pronunciation.PronunciationStore(tmp)
    fails += check(
        "existing dictionary adopted from config",
        store.dictionary_id == config.CARTESIA_PRON_DICT_ID and bool(store.dictionary_id),
        str(store.dictionary_id),
    )

    tts = CartesiaTTSStream()
    tts._ws = FakeWS()
    asyncio.run(tts.send_text("ctx", "నమస్కారం", False))
    payload = tts._ws.sent[0]
    fails += check("tts language te", payload.get("language") == "te", str(payload.get("language")))
    fails += check(
        "tts dictionary passed",
        payload.get("pronunciation_dict_id") == config.CARTESIA_PRON_DICT_ID,
        str(payload.get("pronunciation_dict_id")),
    )
    fails += check(
        "tts speed config",
        payload.get("generation_config", {}).get("speed") == config.CARTESIA_SPEED,
        str(payload.get("generation_config")),
    )

    fails += streaming_check()
    fails += routing_check()

    print(f"\n{'PASS' if not fails else 'FAIL'} ({fails} failure(s))")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
