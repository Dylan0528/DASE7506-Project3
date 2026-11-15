"""Development-only staged ablation: final-choice model and Occam prior."""
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
        "immediate_submit": 0, "choice_bonus": 0.0, "pin_first": 1}


def run(items, bank, configs):
    results = []
    for config in configs:
        for key in agent_module.ENV_KNOBS:
            os.environ.pop("MP3_" + key.upper(), None)
        for key, value in config.items():
            os.environ["MP3_" + key.upper()] = str(value)
        agent_module._SCHEMA_CACHE.clear()
        correct = 0
        calls = 0
        for item in items:
            result = rollout(item, bank, agent_module.create_agent)
            correct += bank.correct(item, result["sql"])
            calls += result["tool_calls"]
        results.append((correct, calls / len(items), config))
    return sorted(results, key=lambda entry: (-entry[0], entry[1]))


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    phase = sys.argv[2] if len(sys.argv) > 2 else "a"
    items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
    bank = Bank(split)
    configs = []
    if phase == "a":
        for choice in ("choice", "choice2", "choice_pair", "choice_pair2",
                       "choice_list", "choice_list2"):
            for bonus in (0.0, 0.15, 0.3, 0.6, 1.0, 2.0):
                configs.append({**BASE, "choice_model": choice, "choice_bonus": bonus})
    elif phase == "b":
        for stop in (2, 3, 4, 5, 9):
            for pin in (0, 1):
                for priority in ("priority2", "priority_pair2"):
                    configs.append({**BASE, "stop_matches": stop,
                                    "stop_matches_empty": stop + 1,
                                    "pin_first": pin, "priority_model": priority})
    elif phase == "c":
        for stop in (2, 3, 4, 5, 9):
            for bonus in (0.0, 0.15, 0.3, 0.6, 1.0):
                for choice in ("choice2", "choice_pair2"):
                    configs.append({**BASE, "stop_matches": stop,
                                    "stop_matches_empty": stop + 1,
                                    "choice_model": choice, "choice_bonus": bonus})
    elif phase == "d":
        for column_bonus in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
            for choice in ("choice2", "choice_pair2"):
                for choice_bonus in (0.0, 0.3, 1.0):
                    configs.append({**BASE, "choice_model": choice,
                                    "choice_bonus": choice_bonus,
                                    "column_bonus": column_bonus})
    elif phase == "e":
        for column_bonus in (0.0, 0.5, 1.0, 2.0, 4.0):
            for stop in (2, 3, 4, 9):
                configs.append({**BASE, "choice_model": "choice2",
                                "choice_bonus": 0.3,
                                "column_bonus": column_bonus,
                                "stop_matches": stop, "stop_matches_empty": stop + 1})
    elif phase == "f":
        for priority_bonus in (0.0, 0.5, 1.0, 2.0, 4.0):
            for static_bonus in (0.0, 1.0, 2.0):
                for required in (0, 1):
                    configs.append({**BASE, "choice_bonus": 0.3,
                                    "column_bonus": 2.0,
                                    "priority_column_bonus": priority_bonus,
                                    "choice_static_bonus": static_bonus,
                                    "schema_columns_required": required})
    elif phase == "g":
        for column_bonus in (0.0, 1.0, 2.0, 4.0):
            for choice_bonus in (0.0, 0.3, 1.0):
                for choice in ("choice2", "choice_pair2"):
                    configs.append({**BASE, "choice_bonus": choice_bonus,
                                    "column_bonus": column_bonus,
                                    "choice_model": choice,
                                    "priority_column_bonus": 1.0,
                                    "choice_static_bonus": 1.0})
    elif phase == "h":
        final = dict(BASE)
        final.update(choice_bonus=0.3, column_bonus=2.0,
                     priority_column_bonus=0.5, choice_static_bonus=0.0,
                     schema_columns_required=0, use_hints=1)
        for stop in (1, 2, 3, 4):
            for limit in (3, 4, 5, 6, 8):
                configs.append({**final, "stop_matches": stop,
                                "stop_matches_empty": stop + 1,
                                "max_checks": limit})
    results = run(items, bank, configs)
    print("split", split, "phase", phase, "n", len(items), "configs", len(results))
    for correct, calls, config in results[:15]:
        print(f"{correct:3d} {100 * correct / len(items):6.2f} calls={calls:.2f} "
              + json.dumps(config, sort_keys=True))
    (OUT / f"grid2-{split}-{phase}.json").write_text(
        json.dumps([{"correct": c, "calls": k, "config": g} for c, k, g in results],
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
