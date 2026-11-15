"""Development-only: decompose remaining failures for an arbitrary configuration."""
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import grid3  # noqa: E402
import submission as agent_module  # noqa: E402
from sim_agent import Bank, rollout  # noqa: E402

OUT = ROOT / "outputs"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    config = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
    bank = Bank(split)
    grid3.configure(config)
    stats = collections.Counter()
    details = []
    for item in items:
        result = rollout(item, bank, agent_module.create_agent)
        passed = bank.correct(item, result["sql"])
        pool = item["pool"]
        public = dict(zip(pool, item["public"]))
        correct = dict(zip(pool, item["correct"]))
        checked = [event["sql"] for event in result["history"] if event["tool"] == "check"]
        correct_visible = [sql for sql in checked if bank.correct(item, sql)]
        has_correct = any(item["correct"])
        stats["tasks"] += 1
        stats["correct"] += passed
        stats["ceiling"] += has_correct
        stats["correct_visible"] += bool(correct_visible)
        stats["no_public_match"] += not any(
            bank.public_match(item, sql) for sql in checked)
        if not passed:
            if not has_correct:
                stats["loss_no_candidate"] += 1
                kind = "no_candidate"
            elif not correct_visible:
                stats["loss_not_found"] += 1
                kind = "not_found"
            else:
                stats["loss_wrong_choice"] += 1
                kind = "wrong_choice"
            details.append({"id": item["id"], "question": item["question"],
                            "kind": kind, "submitted": result["sql"],
                            "correct_candidates": [sql for sql, ok in correct.items()
                                                   if ok][:3],
                            "matching": [sql for sql in checked
                                         if bank.public_match(item, sql)][:8],
                            "checked": checked[:10], "example": item["example"]})
    print(split, json.dumps(dict(stats), indent=2))
    (OUT / f"analyze2-{split}.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in details:
        if row["kind"] != "no_candidate":
            print("---", row["kind"], row["id"], row["question"][:120])
            print(" submitted:", row["submitted"].replace("\n", " ")[:200])
            print(" correct:", [s.replace("\n", " ")[:200]
                               for s in row["correct_candidates"]])


if __name__ == "__main__":
    main()
