import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: acceptance_gate.py <metrics.json>")
        return 2
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        data = json.load(fh)

    rec = data.get("recommendation_latency_ms", {})
    p95 = float(rec.get("p95", 0.0))
    p99 = float(rec.get("p99", 0.0))
    illegal = float(data.get("transitions", {}).get("illegal_transition_reject_rate", 0.0))

    if p95 > 300 or p99 > 700:
        print("FAIL: latency gates")
        return 1
    if illegal > 0.15:
        print("FAIL: transition reject rate")
        return 1

    print("PASS: acceptance gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
