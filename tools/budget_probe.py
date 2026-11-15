"""Development-only: budget/accuracy trade-off probe with the exact simulator.

Reports correct count, mean charged calls and mean checks for a list of configs,
plus the position at which the correct SQL first appears in the check history
under the shipped configuration.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import grid3  # noqa: E402
from sim_agent import Bank, rollout  # noqa: E402
import submission as agent_module  # noqa: E402

OUT = ROOT / "outputs"

FINAL = {
    "dedupe_queue": 1, "dedupe_matches": 1, "wide_candidates": 1,
    "distinct_relative_penalty": 3.0, "like_relative_penalty": 3.0,
}


def probe(config, items, bank, detail=False):
    grid3.configure(config)
    correct = calls = checks = 0
    first_positions = {}
    saved = []
    for item in items:
        result = rollout(item, bank, agent_module.create_agent)
        ok = bank.correct(item, result["sql"])
        correct += ok
        calls += result["tool_calls"]
        checked = [e for e in result["history"] if e["tool"] == "check"]
        checks += len(checked)
        if detail:
            positions = [i + 1 for i, e in enumerate(checked)
                         if bank.correct(item, e["sql"])]
            first_positions[item["id"]] = positions[0] if positions else None
            saved.append({"id": item["id"], "passed": bool(ok),
                          "checks": len(checked),
                          "matches": sum(1 for e in checked
                                         if e["observation"].get("matches_example")),
                          "first_correct": positions[0] if positions else None,
                          "submitted": result["sql"]})
    return {"correct": correct, "calls": round(calls / len(items), 3),
            "checks": round(checks / len(items), 3),
            "first_positions": first_positions, "detail": saved}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    banks = {}
    for split in ("validation", "test"):
        banks[split] = (json.loads((OUT / f"pool-labels-{split}.json").read_text()),
                        Bank(split))
    if mode == "detail":
        items, bank = banks["test"]
        r = probe(FINAL, items, bank, detail=True)
        print("baseline", r["correct"], "calls", r["calls"], "checks", r["checks"])
        pos = [d["first_correct"] for d in r["detail"] if d["passed"]]
        import collections
        print("first-correct-check histogram (passed tasks):",
              sorted(collections.Counter(pos).items(), key=lambda x: (x[0] is None, x[0])))
        print("passed with >=5 checks:",
              sum(1 for d in r["detail"] if d["passed"] and d["checks"] >= 5))
        print("failed with >=5 checks:",
              sum(1 for d in r["detail"] if not d["passed"] and d["checks"] >= 5))
        print("failed with zero matches:",
              sum(1 for d in r["detail"] if not d["passed"] and d["matches"] == 0))
        (OUT / "budget-detail-test.json").write_text(
            json.dumps(r["detail"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    configs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [{}]
    for config in configs:
        merged = {**FINAL, **config}
        row = {"config": config}
        for split, (items, bank) in banks.items():
            r = probe(merged, items, bank)
            row[split] = r["correct"]
            row[split + "_calls"] = r["calls"]
            row[split + "_checks"] = r["checks"]
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
