"""Development-only: decompose remaining failures for a chosen configuration."""
import collections
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

BEST = {"priority_model": "priority2", "choice_model": "choice2",
        "feature_set": "vector2", "stop_matches": 3, "stop_matches_empty": 4,
        "schema_cache": 1, "queue_mode": "model", "max_checks": 8,
        "immediate_submit": 0, "choice_bonus": 0.3, "column_bonus": 2.0,
        "priority_column_bonus": 0.5, "choice_static_bonus": 0.0,
        "pin_first": 1, "use_hints": 1}


def configure(config):
    for key in agent_module.ENV_KNOBS:
        os.environ.pop("MP3_" + key.upper(), None)
    for key, value in config.items():
        os.environ["MP3_" + key.upper()] = str(int(value) if isinstance(value, bool) else value)
    agent_module._SCHEMA_CACHE.clear()


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
    bank = Bank(split)
    configure(BEST)
    stats = collections.Counter()
    details = []
    for item in items:
        result = rollout(item, bank, agent_module.create_agent)
        passed = bank.correct(item, result["sql"])
        pool = item["pool"]
        public = dict(zip(pool, item["public"]))
        correct = dict(zip(pool, item["correct"]))
        checked = [event["sql"] for event in result["history"] if event["tool"] == "check"]
        matching = [sql for sql in checked if public.get(sql, False)]
        correct_visible = [sql for sql in checked if correct.get(sql, False)]
        has_correct = any(item["correct"])
        stats["tasks"] += 1
        stats["correct"] += passed
        stats["ceiling"] += has_correct
        stats["correct_visible"] += bool(correct_visible)
        stats["no_public_match"] += not matching
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
                            "correct_candidates": [sql for sql, ok in correct.items() if ok][:3],
                            "matching": matching[:6], "checked": checked[:8],
                            "example": item["example"]})
    print(split, json.dumps(dict(stats), indent=2))
    (OUT / f"analyze-{split}.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT / f"analyze-{split}.json", len(details))


if __name__ == "__main__":
    main()
