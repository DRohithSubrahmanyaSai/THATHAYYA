"""Fast checks for the live intent router (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.intent import classify

CASES = [
    ("ఎవరు మలజెప్పి నలూడి గారు ఎవరు", 0.92, "unknown_person"),
    ("మలజెప్పి నలూడి గారు ఎవరు", 0.90, "unknown_person"),
    ("అల్లుడు గారి పేరేంటి?", 0.90, "normal"),
    ("నీ భార్య పేరు ఏంటి?", 0.90, "normal"),
    ("చా కాదు చానా", 0.90, "correction"),
    ("డాగ్ కాదు దగ్గు అనాలి", 0.90, "correction"),
    ("గుర్తు పెట్టుకో, మా ఊరు జిన్నూరు", 0.95, "teach"),
    ("హలో తాతయ్య", 0.95, "greeting"),
    ("నీకు టీ కావాలా?", 0.95, "normal"),
    ("నువ్వు ఎవరు?", 0.95, "normal"),
    ("గోరెండ్ బియ్యం అసలు ఉండదు కదండీ పిచ్చి పిచ్చిగా మాట్లాడుతున్నారు", 0.95, "complaint"),
    ("అదే ఏంటి?", 0.95, "repeat"),
    ("మళ్ళీ చెప్పండి", 0.95, "repeat"),
    ("అదే అండి అన్నం తిన్నారా లేదా అని", 0.95, "normal"),
    ("ఏం గోరేంటి?", 0.95, "unclear"),
    ("అదే ఏం గోరు వేసుకొని తిన్నారు అన్నావ్", 0.95, "unclear"),
    ("ఏమో అర్థం కాలేదు", 0.35, "unclear"),
    ("", 0.90, "unclear"),
]


def main() -> int:
    failed = 0
    for text, conf, want in CASES:
        got = classify(text, conf)
        ok = got == want
        failed += 0 if ok else 1
        print(f"[{'OK ' if ok else 'FAIL'}] {got:14s} (want {want:14s}) conf={conf} :: {text}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
