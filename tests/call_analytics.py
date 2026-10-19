"""Call analytics: summarize latency + confidence from runtime logs.

Parses every runtime/logs/call_*.jsonl and prints per-call + aggregate stats:
- first_audio latency (assistant turns)
- total response time (assistant turns)
- STT confidence (user turns)
- turn counts

Run:
    .\.venv\Scripts\python.exe tests/call_analytics.py
"""
import glob
import json
import statistics
from pathlib import Path


def fmt(v) -> str:
    if v is None:
        return "--"
    return f"{v:g}"


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    out = {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 2),
        "median": round(statistics.median(vals), 2),
        "max": round(max(vals), 2),
    }
    if len(vals) >= 2:
        out["p90"] = round(statistics.quantiles(vals, n=10)[-1], 2)
    else:
        out["p90"] = out["max"]
    return out


def parse_float(s: str) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if s.endswith("s"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    logs = sorted(glob.glob("runtime/logs/call_*.jsonl"))
    if not logs:
        print("No call logs found under runtime/logs/")
        return

    all_first_audio: list[float] = []
    all_total: list[float] = []
    all_conf: list[float] = []

    print(f"{'call':28}  turns  fa_mean fa_med fa_max  tot_mean conf_mean")
    print("-" * 72)
    for path in logs:
        name = Path(path).name
        fa, total, conf = [], [], []
        turns = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = rec.get("role")
                turns += 1
                if role == "assistant":
                    fa.append(parse_float(rec.get("first_audio")))
                    total.append(parse_float(rec.get("total")))
                elif role == "user":
                    conf.append(rec.get("confidence"))

        fa = [v for v in fa if v is not None]
        total = [v for v in total if v is not None]
        conf = [v for v in conf if v is not None]

        fa_st = stats(fa)
        tot_st = stats(total)
        conf_st = stats(conf)

        all_first_audio.extend(fa)
        all_total.extend(total)
        all_conf.extend(conf)

        print(
            f"{name:28}  {turns:4d}  "
            f"{fmt(fa_st['mean']):>6} {fmt(fa_st['median']):>5} {fmt(fa_st['max']):>6}  "
            f"{fmt(tot_st['mean']):>8} {fmt(conf_st['mean']):>7}"
        )

    print("-" * 72)
    fa_st = stats(all_first_audio)
    tot_st = stats(all_total)
    conf_st = stats(all_conf)
    print(
        f"{'AGGREGATE':28}  {len(logs)} calls  "
        f"{fmt(fa_st['mean']):>6} {fmt(fa_st['median']):>5} {fmt(fa_st['max']):>6}  "
        f"{fmt(tot_st['mean']):>8} {fmt(conf_st['mean']):>7}"
    )
    print(f"first_audio n={fa_st['n']}  total n={tot_st['n']}  confidence n={conf_st['n']}")


if __name__ == "__main__":
    main()
