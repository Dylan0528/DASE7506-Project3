"""Development-only: list tasks that flip between two configurations."""
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


def run(split, config):
    items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
    bank = Bank(split)
    grid3.configure(config)
    result = {}
    for item in items:
        outcome = rollout(item, bank, agent_module.create_agent)
        result[item["id"]] = (bool(bank.correct(item, outcome["sql"])),
                              outcome["sql"], item["question"], outcome["tool_calls"])
    return result, items


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    base = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    other = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    a, items = run(split, base)
    b, _ = run(split, other)
    gains = [key for key in a if b[key][0] and not a[key][0]]
    losses = [key for key in a if a[key][0] and not b[key][0]]
    print(split, "base", sum(v[0] for v in a.values()),
          "other", sum(v[0] for v in b.values()),
          "gains", len(gains), "losses", len(losses))
    by_id = {item["id"]: item for item in items}
    for label, keys in (("GAIN", gains), ("LOSS", losses)):
        for key in keys:
            item = by_id[key]
            winners = [sql for sql, ok in zip(item["pool"], item["correct"]) if ok]
            print("---", label, key, "calls", a[key][3], "->", b[key][3])
            print(" Q:", item["question"][:180])
            print(" base:", a[key][1].replace("\n", " ")[:220])
            print(" other:", b[key][1].replace("\n", " ")[:220])
            print(" correct-in-pool:", [sql.replace("\n", " ")[:200] for sql in winners[:2]])


if __name__ == "__main__":
    main()
