"""Development-only: compare many configs in one pass and report per-task flips."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import grid3  # noqa: E402
import submission as agent_module  # noqa: E402
from sim_agent import Bank, rollout  # noqa: E402

OUT = ROOT / "outputs"


def run(config, items, bank):
    grid3.configure(config)
    out = {}
    for item in items:
        r = rollout(item, bank, agent_module.create_agent)
        out[item["id"]] = {"ok": bool(bank.correct(item, r["sql"])),
                           "sql": r["sql"], "calls": r["tool_calls"],
                           "checks": sum(1 for e in r["history"] if e["tool"] == "check")}
    return out


def main():
    configs = json.loads(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
    bank = Bank(split)
    by_id = {x["id"]: x for x in items}
    results = []
    for config in configs:
        r = run(config, items, bank)
        results.append((config, r))
        print(json.dumps({"correct": sum(v["ok"] for v in r.values()),
                          "calls": round(sum(v["calls"] for v in r.values()) / len(items), 3),
                          "checks": round(sum(v["checks"] for v in r.values()) / len(items), 3),
                          "config": config}, ensure_ascii=False), flush=True)
    base_config, base = results[0]
    print("=== flips vs first config ===")
    for config, r in results[1:]:
        gains = [k for k in base if r[k]["ok"] and not base[k]["ok"]]
        losses = [k for k in base if base[k]["ok"] and not r[k]["ok"]]
        print("CONFIG", json.dumps(config, ensure_ascii=False),
              "gains", len(gains), "losses", len(losses))
        for label, keys in (("GAIN", gains), ("LOSS", losses)):
            for k in keys:
                print(" ", label, k, "| Q:", by_id[k]["question"][:110])
                print("    base:", base[k]["sql"].replace("\n", " ")[:170])
                print("    new :", r[k]["sql"].replace("\n", " ")[:170])
    (OUT / "combo-probe.json").write_text(
        json.dumps({"split": split,
                    "results": [{"config": c, "correct": sum(v["ok"] for v in r.values())}
                                for c, r in results]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
