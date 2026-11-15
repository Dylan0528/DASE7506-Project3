"""Fit candidate ranking/choice models on the training split only.

The feature distribution matches the exported agent's candidate generator exactly.
"""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ranker import vector, vector2  # noqa: E402
from train_ranker import fit, fit_pairwise  # noqa: E402

OUT = ROOT / "outputs"


def collect(labels, feature_fn):
    samples, groups, all_groups = [], [], []
    for item in labels:
        task = []
        first = item["suggestions"][0] if item["suggestions"] else item["issue_sql"]
        rows = len(item["example"]["rows"])
        for index, (sql, public, correct) in enumerate(
                zip(item["pool"], item["public"], item["correct"]), start=1):
            if feature_fn is vector2:
                features = feature_fn(sql, item["question"], item["schema"],
                                      item["column_count"], item["issue_sql"],
                                      index, first, rows)
            else:
                features = feature_fn(sql, item["question"], item["schema"],
                                      item["column_count"], item["issue_sql"], index)
            sample = (sql, features, {"public": public, "correct": correct},
                      3.0 if correct else 1.0)
            samples.append(sample)
            task.append(sample)
        groups.append([sample for sample in task if sample[2]["public"]])
        all_groups.append(task)
    return samples, groups, all_groups


def fit_listwise(groups, epochs=80, learning_rate=0.18, l2=0.0006):
    """Softmax over public-matching candidates: pick a 3-instance-correct one."""
    flat = [sample for group in groups for sample in group]
    keys = sorted({key for sample in flat for key in sample[1]})
    means = {key: sum(sample[1].get(key, 0.0) for sample in flat) / len(flat)
             for key in keys}
    scales = {}
    for key in keys:
        variance = sum((sample[1].get(key, 0.0) - means[key]) ** 2
                       for sample in flat) / len(flat)
        scales[key] = max(math.sqrt(variance), 0.05)
    active = [group for group in groups if any(sample[2]["correct"] for sample in group)]
    if not active:
        return {"keys": keys, "means": means, "scales": scales,
                "coefficients": {key: 0.0 for key in keys}, "intercept": 0.0}
    rows = [[[(sample[1].get(key, 0.0) - means[key]) / scales[key] for key in keys]
             for sample in group] for group in active]
    coefficients = {key: 0.0 for key in keys}
    for _ in range(epochs):
        gradient = {key: 0.0 for key in keys}
        for group, vectors in zip(active, rows):
            scores = [sum(value * coefficients[key] for key, value in zip(keys, row))
                      for row in vectors]
            top = max(scores)
            weights = [math.exp(max(min(score - top, 30), -30)) for score in scores]
            total = sum(weights)
            positives = sum(sample[2]["correct"] for sample in group)
            targets = [1.0 / positives if sample[2]["correct"] else 0.0
                       for sample in group]
            for row, weight, target in zip(vectors, weights, targets):
                error = weight / total - target
                for key, value in zip(keys, row):
                    gradient[key] += error * value
        for key in keys:
            coefficients[key] -= (learning_rate * gradient[key] / len(active) +
                                  l2 * coefficients[key])
    return {"keys": keys, "means": means, "scales": scales,
            "coefficients": coefficients, "intercept": 0.0}


def main():
    labels = json.loads((OUT / "pool-labels-train.json").read_text())
    result = {
        "provenance": "trained on mp3_sql/data/train_references.json, train_memory hints, "
                      "and the exported candidate generator (train split only)",
        "split": "train", "tasks": len(labels),
        "samples": sum(len(item["pool"]) for item in labels),
        "config": {"priority_model": "priority2", "choice_model": "choice2",
                   "feature_set": "vector2", "stop_matches": 3,
                   "stop_matches_empty": 4, "schema_cache": True,
                   "queue_mode": "model", "max_checks": 8,
                   "immediate_submit": False, "choice_bonus": 0.3,
                   "column_bonus": 2.0, "priority_column_bonus": 0.5,
                   "choice_static_bonus": 0.0, "pin_first": True,
                   "schema_columns_required": 0},
    }
    for name, fn in (("v1", vector), ("v2", vector2)):
        samples, groups, all_groups = collect(labels, fn)
        suffix = "" if name == "v1" else "2"
        result["priority" + suffix] = fit(samples, "correct")
        result["choice" + suffix] = fit([s for s in samples if s[2]["public"]], "correct")
        result["priority_pair" + suffix] = fit_pairwise(all_groups)
        result["choice_pair" + suffix] = fit_pairwise(groups)
        result["choice_list" + suffix] = fit_listwise(groups)
        print(name, "samples", len(samples), "public", sum(s[2]["public"] for s in samples),
              "correct", sum(s[2]["correct"] for s in samples))
    # v1 keys intentionally keep the plain names so old ablations stay reproducible.
    path = ROOT / "submission" / "ranker.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
