import time
import argparse

from . import config, mic
from .brain import Brain
from .memory import CallLog
from .stt import STT
from .tts import TTS
from .vad import SileroVad


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["indicf5_local", "cartesia"], default=config.TTS_LIVE_ENGINE)
    p.add_argument("--stt", choices=["faster-whisper", "deepgram"], default=config.STT_ENGINE)
    p.add_argument("--model", help="Groq model override")
    p.add_argument("--mute", action="store_true", help="Disable audio playback")
    return p.parse_args()


def main():
    args = parse_args()
    if args.engine:
        config.TTS_LIVE_ENGINE = args.engine
    if args.stt:
        config.STT_ENGINE = args.stt
    if args.model:
        config.GROQ_MODEL = args.model

    print("Loading brain (Groq)...")
    brain = Brain()
    print(f"Loading TTS ({config.TTS_LIVE_ENGINE})...")
    tts = TTS(config.TTS_LIVE_ENGINE)
    print(f"Loading STT ({config.STT_ENGINE})...")
    stt = STT()
    print("Loading VAD (Silero)...")
    vad = SileroVad()
    log = CallLog()
    print(f"learned memories: {brain.memory.count()}  |  log: {log.path.name}")
    print("\nReady. Speak into your mic. Ctrl+C to quit.\n")

    while True:
        try:
            audio, sr = mic.listen_until_silence(vad)
        except KeyboardInterrupt:
            break
        if audio is None or len(audio) < sr // 2:
            continue

        t0 = time.time()
        try:
            text = stt.transcribe(audio, sr)
        except Exception as e:
            print(f"STT error: {e}")
            continue
        if not text.strip():
            continue
        print(f"you       : {text}")
        log.write("user", text)

        t1 = time.time()
        try:
            reply = brain.chat(text)
        except Exception as e:
            print(f"LLM error: {e}")
            continue
        t2 = time.time()
        print(f"Thathayya : {reply}")

        try:
            wav, out_sr = tts.synthesize(reply)
            if not args.mute:
                mic.play(wav, out_sr)
        except Exception as e:
            print(f"TTS error: {e}")

        print(f"[latency] stt={t1 - t0:.2f}s  llm={t2 - t1:.2f}s  total={time.time() - t0:.2f}s\n")
        log.write(
            "assistant",
            reply,
            stt=round(t1 - t0, 2),
            llm=round(t2 - t1, 2),
            total=round(time.time() - t0, 2),
        )

    print("bye")


if __name__ == "__main__":
    main()
