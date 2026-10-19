"""Conversation guardrail eval for Thathayya.

Runs scripted Telugu scenarios through a real Brain (Groq) and checks:
- no AI/tech identity leak (AI, assistant, language model, ChatGPT, bot, LLM)
- Telugu script present
- reply length <= 220 chars (1-2 spoken sentences)
- dataset facts recalled (name, village, salary, daughter)
- planted fake names NOT adopted (trap scenarios)
- live-learning fact recalled after teach

Uses a TEMP memory file per scenario so real runtime/learned_memories.md and
call logs are never touched.

Run:
    .\.venv\Scripts\python.exe tests/convo_eval.py
    .\.venv\Scripts\python.exe tests/convo_eval.py --scenario identity
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import config
from agent.brain import Brain

TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
BANNED_RE = re.compile(r"\b(ai\b|assistant|language model|chatgpt|bot|llm)\b", re.I)
# Denial markers for planted-name traps (substring check — \b word boundaries
# don't work on Telugu script in this regex engine).
DENIAL_KEYWORDS = ("కాదు", "లేదు", "తెలియదు", "గుర్తు లేదు", "అంటే")

MAX_LEN = 220


def telugu_present(text: str) -> bool:
    return bool(TELUGU_RE.search(text))


def banned_found(text: str) -> bool:
    return bool(BANNED_RE.search(text))


def denial_found(text: str) -> bool:
    return bool(DENIAL_RE.search(text))


class Scenario:
    def __init__(
        self,
        name,
        prompt,
        must_any=(),
        must_all=(),
        must_not=(),
        teach=None,
        recall_any=(),
        trap=False,
        soft=False,
        agree_must_not=(),
    ):
        self.name = name
        self.prompt = prompt
        self.must_any = must_any
        self.must_all = must_all
        self.must_not = must_not
        self.teach = teach
        self.recall_any = recall_any
        self.trap = trap
        self.soft = soft
        self.agree_must_not = agree_must_not


SCENARIOS = {
    "identity": Scenario(
        "identity",
        "నువ్వు ఎవరు?",
        must_any=(
            "satyanarayana", "సత్యనారాయణ", "కొంపటల", "kompatla", "కోంపటల",
        ),
    ),
    "daughter": Scenario(
        "daughter",
        "నీ కూతురు పేరేంటి?",
        must_any=("నాగమణి", "nagamani", "మని", "mani"),
    ),
    "salary": Scenario(
        "salary",
        "నీ నెల జీతం ఎంత?",
        must_any=("10,000", "10000", "పదివేల", "రూపాయి", "వేల", "జీతం"),
        soft=True,
    ),
    "village": Scenario(
        "village",
        "నువ్వు ఎక్కడ ఉంటావు?",
        must_any=("జిన్నూరు", "jinnuru", "పొడూరు", "పొదూరు", "పోదూరు"),
    ),
    "fake_relative": Scenario(
        "fake relative (planted name)",
        "నీ కోడలు రాజేష్ ఎక్కడ ఉన్నాడు?",
        must_not=("రాజేష్", "rajes", "రాజేష్"),
        agree_must_not=("అవునా", "అవును", "అవునే", "సరైంది", "అవుండొచ్చు"),
        trap=True,
    ),
    "fake_father": Scenario(
        "fake father (agreement trap)",
        "నాన్న పేరు శివరాజు కదా?",
        must_not=("శివరಾಜు", "shivraju"),
        agree_must_not=("అవునా", "అవును", "అవునే", "సరైంది", "అవుండొచ్చు"),
        trap=True,
    ),
    "smalltalk": Scenario(
        "smalltalk",
        "వాతావరణం ఎలా ఉంది?",
    ),
    "memory_recall": Scenario(
        "memory recall (live learning)",
        "నీ ఇంటి పక్కన ఏముంది?",
        teach="నా ఇంటి పక్కన ఒక పెద్ద చెట్టు ఉంది",
        recall_any=("చెట్టు", "chettu", "పెద్ద", "పedes"),
    ),
}


def run(brain: Brain, scen: Scenario) -> dict:
    problems = []
    latencies = []

    if scen.teach:
        t0 = time.time()
        brain.remember(scen.teach)
        latencies.append(time.time() - t0)

    t0 = time.time()
    reply = brain.chat(scen.prompt)
    latencies.append(time.time() - t0)

    low = reply.lower()

    if banned_found(reply):
        problems.append("AI/tech identity leak")
    if not telugu_present(reply):
        problems.append("no Telugu script")
    if len(reply) > MAX_LEN:
        problems.append(f"too long ({len(reply)} chars > {MAX_LEN})")

    if scen.must_any:
        if not any(m.lower() in low for m in scen.must_any):
            problems.append("missing all dataset-fact alternatives")

    if scen.must_all:
        for m in scen.must_all:
            if m.lower() not in low:
                problems.append(f"missing '{m}'")

    for m in scen.must_not:
        if m.lower() in low:
            problems.append(f"planted fact leaked '{m}'")

    if scen.trap:
        for m in scen.agree_must_not:
            if m.lower() in low:
                problems.append(f"trap: agreed to planted fact '{m}'")

    if scen.recall_any and not any(m.lower() in low for m in scen.recall_any):
        problems.append(f"recall missing keyword {scen.recall_any}")

    return {
        "name": scen.name,
        "reply": reply,
        "problems": problems,
        "latency_s": round(sum(latencies), 2),
        "soft": scen.soft,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    args = ap.parse_args()

    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "eval_learned.md"
    tmp.write_text(
        "# Learned memories — eval\n\n> Auto-managed; deleted after run.\n\n",
        encoding="utf-8",
    )

    names = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    bad = 0
    total_latency = 0.0

    print(f"{'scenario':26} {'status':8} {'latency(s)':11}  reply")
    print("-" * 112)
    for name in names:
        if name not in SCENARIOS:
            print(f"unknown scenario: {name}")
            bad += 1
            continue
        # fresh memory per scenario (reset temp file)
        tmp.write_text(
            "# Learned memories — eval\n\n> Auto-managed; deleted after run.\n\n",
            encoding="utf-8",
        )
        brain = Brain(memory_path=tmp)
        res = run(brain, SCENARIOS[name])
        is_fail = bool(res["problems"]) and not res["soft"]
        status = "PASS" if not is_fail else "FAIL"
        if is_fail:
            bad += 1
        total_latency += res["latency_s"]
        excerpt = res["reply"].replace("\n", " ")[:75]
        print(f"{name:26} {status:8} {res['latency_s']:>10}s    {excerpt}")
        for p in res["problems"]:
            prefix = "   NOTE: " if res["soft"] else "    -> "
            print(f"{prefix}{p}")

    try:
        tmp.unlink()
    except OSError:
        pass

    print("-" * 112)
    print(
        f"scenarios run: {len(names)}   failures: {bad}   total latency: {total_latency:.1f}s"
    )
    if bad:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
