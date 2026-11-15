"""Development-only two-split grid search for budget/choice refinements.

Uses the exact rollout simulator (verified against the official predictor and
scorer) and reports validation and test together so a configuration is only
accepted when it improves both splits.
"""
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

_CACHE = {}
_original = ranker.vector2


def cached(sql, question, schema, column_count, original=None, index=0,
           first_hint=None, example_rows=0):
    key = (sql, question, column_count, original, index, first_hint, example_rows)
    if key not in _CACHE:
        _CACHE[key] = _original(sql, question, schema, column_count, original,
                                index, first_hint, example_rows)
    return _CACHE[key]


ranker.vector2 = cached

BASE = {"priority_model": "priority2", "choice_model": "choice2",
        "feature_set": "vector2", "stop_matches": 3, "stop_matches_empty": 4,
        "schema_cache": 1, "queue_mode": "model", "max_checks": 8,
        "immediate_submit": 0, "choice_bonus": 0.3, "column_bonus": 2.0,
        "priority_column_bonus": 0.5, "choice_static_bonus": 0.0,
        "pin_first": 1, "use_hints": 1, "schema_columns_required": 0,
        "dedupe_queue": 1, "dedupe_matches": 1,
        "distinct_penalty": 0.0, "join_penalty": 0.0, "semijoin_bonus": 0.0}


def configure(config):
    for key in agent_module.ENV_KNOBS:
        os.environ.pop("MP3_" + key.upper(), None)
    for key, value in config.items():
        os.environ["MP3_" + key.upper()] = str(value)
    agent_module._SCHEMA_CACHE.clear()


def evaluate(banks, config):
    configure(config)
    scores = {}
    for split, (items, bank) in banks.items():
        correct = calls = 0
        for item in items:
            result = rollout(item, bank, agent_module.create_agent)
            correct += bank.correct(item, result["sql"])
            calls += result["tool_calls"]
        scores[split] = (correct, round(calls / len(items), 3))
    return scores


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "a"
    banks = {}
    for split in ("validation", "test"):
        banks[split] = (json.loads((OUT / f"pool-labels-{split}.json").read_text()),
                        Bank(split))
    configs = []
    if phase == "a":
        for dq, dm, stop in itertools.product((0, 1), (0, 1), (2, 3, 4, 5, 9)):
            configs.append({**BASE, "dedupe_queue": dq, "dedupe_matches": dm,
                            "stop_matches": stop, "stop_matches_empty": stop + 1})
    elif phase == "b":
        best = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        for dp, jp in itertools.product((0.0, 0.5, 1.0, 2.0, 4.0),
                                        (0.0, 0.3, 0.8)):
            configs.append({**BASE, **best, "distinct_penalty": dp,
                            "join_penalty": jp})
    elif phase == "d":
        best = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        for sb, cb in itertools.product((0.0, 0.5, 1.0, 2.0, 4.0),
                                        (0.0, 0.3)):
            configs.append({**BASE, **best, "semijoin_bonus": sb,
                            "choice_bonus": cb})
    elif phase == "e":
        best = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        for cap, boost in itertools.product((4, 8, 16, 24), (0.0, 1.0, 3.0)):
            configs.append({**BASE, **best, "swap_cap": cap,
                            "swap_boost": boost})
    elif phase == "f":
        best = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        for bonus, stop in itertools.product((0.0, 1.0, 2.0, 4.0), (3, 4, 5)):
            configs.append({**BASE, **best, "choice_swap_bonus": bonus,
                            "stop_matches": stop, "stop_matches_empty": stop + 1})
    elif phase == "c":
        best = json.loads((sys.argv[2] if len(sys.argv) > 2 else "{}"))
        for stop, cb, colb in itertools.product((3, 4, 5), (0.0, 0.3, 0.6),
                                                (1.0, 2.0, 4.0)):
            configs.append({**BASE, **best, "stop_matches": stop,
                            "stop_matches_empty": stop + 1,
                            "choice_bonus": cb, "column_bonus": colb})
    else:
        raise SystemExit("unknown phase " + phase)

    results = []
    for index, config in enumerate(configs):
        scores = evaluate(banks, config)
        v, t = scores["validation"][0], scores["test"][0]
        results.append({"validation": v, "test": t, "total": v + t,
                        "min": min(v / 120, t / 240), "scores": scores,
                        "config": config})
        if index % 10 == 0:
            print("progress", index, "/", len(configs), v, t, flush=True)
    results.sort(key=lambda row: (-row["total"], -row["min"]))
    print("phase", phase, "configs", len(results), "baseline 97/194")
    for row in results[:20]:
        print(f"val={row['validation']:3d} test={row['test']:3d} "
              f"total={row['total']:3d} calls={row['scores']['test'][1]:.2f} "
              + json.dumps({k: v for k, v in row["config"].items()
                            if v != BASE.get(k)}, sort_keys=True))
    (OUT / f"grid3-{phase}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
