import argparse
import statistics
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groq import Groq

from agent import config

SYSTEM = "Reply as Thathayya, a warm elder, in spoken Telugu (West Godavari dialect). 1-2 short sentences."
PROMPTS = [
    "నమస్కారం తాతయ్య, ఎలా ఉన్నావ్?",
    "మీ ఊరు గురించి చెప్పు తాతయ్య.",
    "ఈరోజు భోజనం ఏం చేసుకున్నావ్?",
]
DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
]


def bench(client, model, reasoning_effort=None, trials=1):
    ttfts, totals, chars = [], [], 0
    for _ in range(trials):
        for prompt in PROMPTS:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ]
            kwargs = dict(model=model, messages=msgs, max_tokens=200, temperature=0.7, stream=True)
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            t0 = time.time()
            ttft = None
            n = 0
            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    if ttft is None:
                        ttft = time.time() - t0
                    n += len(delta)
            total = time.time() - t0
            ttfts.append(ttft if ttft is not None else total)
            totals.append(total)
            chars += n
    return ttfts, totals, chars


def main():
    p = argparse.ArgumentParser(description="Benchmark free Groq models for voice-agent latency")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)
    args = p.parse_args()

    if not config.GROQ_API_KEY:
        print("GROQ_API_KEY not set")
        return
    client = Groq(api_key=config.GROQ_API_KEY)

    print(f"{'model':26s} {'TTFT(s)':>9} {'total(s)':>9} {'chars':>7}")
    print("-" * 56)
    for model in args.models:
        try:
            ttfts, totals, chars = bench(client, model, reasoning_effort=args.reasoning_effort)
        except Exception as e:  # noqa: BLE001
            print(f"{model:26s} ERROR: {e}")
            continue
        print(
            f"{model:26s} {statistics.mean(ttfts):9.2f} {statistics.mean(totals):9.2f} {chars:7d}"
        )


if __name__ == "__main__":
    main()
