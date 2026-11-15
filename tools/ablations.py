"""Development-only one-factor-at-a-time ablations for the project report."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import submission as agent_module  # noqa: E402
from sim_agent import Bank, rollout  # noqa: E402

OUT = ROOT / "outputs"

FINAL = {
    "priority_model": "priority2", "choice_model": "choice2",
    "feature_set": "vector2", "stop_matches": 1, "stop_matches_empty": 4,
    "min_checks": 2,
    "schema_cache": 1, "queue_mode": "model", "max_checks": 8,
    "immediate_submit": 0, "choice_bonus": 0.3, "column_bonus": 2.0,
    "priority_column_bonus": 0.5, "choice_static_bonus": 0.0,
    "pin_first": 1, "use_hints": 1, "schema_columns_required": 0,
    "dedupe_queue": 1, "dedupe_matches": 1, "wide_candidates": 1,
    "distinct_relative_penalty": 3.0, "like_relative_penalty": 3.0,
}

VARIANTS = {
    "final": {},
    "no_schema_memory": {"schema_cache": 0},
    "heuristic_queue": {"queue_mode": "preferred"},
    "no_column_priors": {"column_bonus": 0.0, "priority_column_bonus": 0.0,
                         "choice_static_bonus": 0.0},
    "no_occam_prior": {"choice_bonus": 0.0},
    "v1_features_only": {"feature_set": "vector", "priority_model": "priority",
                         "choice_model": "choice", "column_bonus": 0.0,
                         "priority_column_bonus": 0.0, "choice_bonus": 0.0},
    "no_hints": {"use_hints": 0},
    "no_stop_early": {"stop_matches": 9, "stop_matches_empty": 10},
    "no_queue_dedup": {"dedupe_queue": 0, "dedupe_matches": 0},
    "no_wide_candidates": {"wide_candidates": 0},
    "no_relative_choice": {"distinct_relative_penalty": 0.0,
                           "like_relative_penalty": 0.0},
    "no_budget_tuning": {"stop_matches": 3, "min_checks": 0},
}


def configure(config):
    for key in agent_module.ENV_KNOBS:
        os.environ.pop("MP3_" + key.upper(), None)
    for key, value in config.items():
        os.environ["MP3_" + key.upper()] = str(value)
    agent_module._SCHEMA_CACHE.clear()


def main():
    output = {}
    for split in ("validation", "test"):
        items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
        bank = Bank(split)
        results = {}
        for name, changes in VARIANTS.items():
            config = {**FINAL, **changes}
            configure(config)
            correct = 0
            calls = 0
            for item in items:
                result = rollout(item, bank, agent_module.create_agent)
                correct += bank.correct(item, result["sql"])
                calls += result["tool_calls"]
            results[name] = {"correct": correct, "n": len(items),
                             "accuracy": round(100 * correct / len(items), 2),
                             "mean_tool_calls": round(calls / len(items), 3)}
        output[split] = results
        print("===", split)
        for name, value in results.items():
            print(f"  {name:20s} {value['correct']:3d}/{value['n']} "
                  f"{value['accuracy']:6.2f}%  calls={value['mean_tool_calls']:.3f}")
    (OUT / "ablations.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
