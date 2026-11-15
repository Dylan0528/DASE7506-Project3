"""Development-only controlled ablations over agent configuration knobs."""
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ranker  # noqa: E402
import submission as agent_module  # noqa: E402
from sim_agent import Bank, rollout  # noqa: E402

OUT = ROOT / "outputs"

# Feature vectors are identical across configurations, so memoize them.
_FEATURE_CACHE = {}
_original_v2 = ranker.vector2
_original_v1 = ranker.vector


def _cached(fn):
    def wrapped(sql, question, schema, column_count, original=None, index=0,
                first_hint=None, example_rows=0):
        key = (sql, question, column_count, original, index, first_hint, example_rows)
        if key not in _FEATURE_CACHE:
            _FEATURE_CACHE[key] = fn(sql, question, schema, column_count, original,
                                     index, first_hint, example_rows)
        return _FEATURE_CACHE[key]
    return wrapped


ranker.vector2 = _cached(_original_v2)
ranker.vector = ranker.vector  # v1 is cheap; leave it alone


def evaluate(items, bank, config):
    for key in agent_module.ENV_KNOBS:
        os.environ.pop("MP3_" + key.upper(), None)
    for key, value in config.items():
        os.environ["MP3_" + key.upper()] = str(int(value) if isinstance(value, bool) else value)
    agent_module._SCHEMA_CACHE.clear()
    correct = 0
    calls = 0
    for item in items:
        result = rollout(item, bank, agent_module.create_agent)
        correct += bank.correct(item, result["sql"])
        calls += result["tool_calls"]
    return correct, calls / len(items)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
    bank = Bank(split)
    configs = []
    base = {"stop_matches": 3, "stop_matches_empty": 4, "schema_cache": 1,
            "queue_mode": "preferred", "priority_model": "priority2",
            "choice_model": "choice2", "max_checks": 8, "immediate_submit": 0}
    for stop in (2, 3, 4, 5, 9):
        for choice in ("choice2", "choice_pair2", "choice_list2", "choice", "choice_pair"):
            configs.append({**base, "stop_matches": stop, "stop_matches_empty": stop + 1,
                            "choice_model": choice})
    for priority in ("priority2", "priority", "priority_pair2", "priority_pair", "choice2"):
        for mode in ("preferred", "model"):
            configs.append({**base, "stop_matches": 3, "stop_matches_empty": 4,
                            "priority_model": priority, "queue_mode": mode})
    for cache in (0, 1):
        for immediate in (0, 1):
            configs.append({**base, "stop_matches": 3, "stop_matches_empty": 4,
                            "schema_cache": cache, "immediate_submit": immediate})
    seen = set()
    results = []
    for config in configs:
        identity = json.dumps(config, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        correct, calls = evaluate(items, bank, config)
        results.append((correct, calls, config))
    results.sort(key=lambda item: (-item[0], item[1]))
    print("split", split, "n", len(items), "configs", len(results))
    for correct, calls, config in results[:20]:
        print(f"{correct:3d} {100 * correct / len(items):6.2f} calls={calls:.2f} "
              + json.dumps(config, sort_keys=True))
    (OUT / f"grid-{split}.json").write_text(
        json.dumps([{"correct": c, "calls": k, "config": g} for c, k, g in results],
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
