"""Development-only: evaluate a small list of configurations on both splits."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import grid3  # noqa: E402
from sim_agent import Bank  # noqa: E402

OUT = ROOT / "outputs"


def main():
    payload = json.loads(sys.argv[1])
    configs = payload if isinstance(payload, list) else [payload]
    splits = ("validation", "test")
    banks = {split: (json.loads((OUT / f"pool-labels-{split}.json").read_text()),
                     Bank(split)) for split in splits}
    for config in configs:
        scores = grid3.evaluate(banks, config)
        print(json.dumps({"val": scores["validation"][0],
                          "test": scores["test"][0],
                          "calls": scores["test"][1], "config": config},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
