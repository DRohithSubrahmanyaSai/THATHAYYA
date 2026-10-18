import argparse

from agent import config
from agent.pipeline_stream import main


def cli():
    p = argparse.ArgumentParser(description="Thathayya realtime (streaming) voice agent")
    p.add_argument("--model", help="Groq model override")
    p.add_argument("--mute", action="store_true", help="Disable audio playback")
    p.add_argument("--half-duplex", action="store_true", help="Disable barge-in (safer on speakers)")
    args = p.parse_args()
    if args.half_duplex:
        config.BARGE_IN = False
    main(model=args.model, mute=args.mute)


if __name__ == "__main__":
    cli()
