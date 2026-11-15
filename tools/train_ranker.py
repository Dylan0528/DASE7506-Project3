"""Fit the exported candidate ranker using only the released training split."""
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))

from mp3_sql.common import load_bank, result_fingerprint  # noqa: E402
from mp3_sql.sqlite_tool import validate_sql  # noqa: E402
from ranker import vector  # noqa: E402
from sql_repairs import candidates  # noqa: E402

DATA = ROOT / "mp3_sql" / "data"


def local_sql(path, sql):
    try:
        validate_sql(sql)
    except Exception:
        return None
    deadline = time.monotonic() + 0.4
    try:
        uri = Path(path).absolute().as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
            cursor = connection.execute(sql)
            raw = cursor.fetchmany(5001)
            if len(raw) > 5000:
                return None
            rows = []
            for row in raw:
                cells = []
                for cell in row:
                    if isinstance(cell, bytes):
                        cell = "blob:" + cell.hex()
                    elif isinstance(cell, float) and not math.isfinite(cell):
                        return None
                    elif isinstance(cell, str) and len(cell.encode()) > 4000:
                        return None
                    cells.append(cell)
                rows.append(cells)
            return {"ok": True, "columns": [item[0] for item in cursor.description],
                    "rows": rows, "error": None}
    except Exception:
        return None


def fit(samples, label_key, epochs=35, learning_rate=0.22, l2=0.0015):
    keys = sorted({key for sample in samples for key in sample[1]})
    means = {key: sum(sample[1].get(key, 0.0) for sample in samples) / len(samples) for key in keys}
    scales = {}
    for key in keys:
        variance = sum((sample[1].get(key, 0.0) - means[key]) ** 2 for sample in samples) / len(samples)
        scales[key] = max(math.sqrt(variance), 0.05)
    rows = [[(sample[1].get(key, 0.0) - means[key]) / scales[key] for key in keys]
            for sample in samples]
    labels = [1.0 if sample[2][label_key] else 0.0 for sample in samples]
    weights = [sample[3] for sample in samples]
    positive = sum(weight for label, weight in zip(labels, weights) if label)
    total = sum(weights)
    coefficients = {key: 0.0 for key in keys}
    intercept = math.log(max(positive, 1) / max(total - positive, 1))
    weight_sum = sum(weights)
    for _ in range(epochs):
        gradient = {key: 0.0 for key in keys}
        bias_gradient = 0.0
        for row, label, weight in zip(rows, labels, weights):
            z = intercept + sum(value * coefficients[key] for key, value in zip(keys, row))
            z = max(min(z, 30), -30)
            error = (1 / (1 + math.exp(-z)) - label) * weight
            bias_gradient += error
            for key, value in zip(keys, row):
                gradient[key] += error * value
        intercept -= learning_rate * bias_gradient / weight_sum
        for key in keys:
            coefficients[key] -= learning_rate * (gradient[key] / weight_sum +
                                                 l2 * coefficients[key])
    return {"keys": keys, "means": means, "scales": scales,
            "coefficients": coefficients, "intercept": intercept}


def fit_pairwise(groups, epochs=70, learning_rate=0.18, l2=0.0008):
    """Rank a correct candidate above public-only false positives of the same task."""
    flat = [sample for group in groups for sample in group]
    keys = sorted({key for sample in flat for key in sample[1]})
    means = {key: sum(sample[1].get(key, 0.0) for sample in flat) / len(flat) for key in keys}
    scales = {}
    for key in keys:
        variance = sum((sample[1].get(key, 0.0) - means[key]) ** 2 for sample in flat) / len(flat)
        scales[key] = max(math.sqrt(variance), 0.05)
    pairs = []
    for group in groups:
        positive = [sample[1] for sample in group if sample[2]["correct"]]
        negative = [sample[1] for sample in group if not sample[2]["correct"]]
        for good in positive:
            for bad in negative:
                difference = [(good.get(key, 0.0) - bad.get(key, 0.0)) / scales[key] for key in keys]
                pairs.append(difference)
    if not pairs:
        return {"keys": keys, "means": means, "scales": scales,
                "coefficients": {key: 0.0 for key in keys}, "intercept": 0.0}
    coefficients = {key: 0.0 for key in keys}
    for _ in range(epochs):
        gradient = {key: 0.0 for key in keys}
        for difference in pairs:
            margin = sum(value * coefficients[key] for key, value in zip(keys, difference))
            margin = max(min(margin, 30), -30)
            error = 1 / (1 + math.exp(-margin)) - 1
            for key, value in zip(keys, difference):
                gradient[key] += error * value
        for key in keys:
            coefficients[key] -= learning_rate * (gradient[key] / len(pairs) +
                                                 l2 * coefficients[key])
    return {"keys": keys, "means": means, "scales": scales,
            "coefficients": coefficients, "intercept": 0.0}


def main():
    rows, manifest, folder = load_bank("train", DATA)
    references = json.loads((folder / "train_references.json").read_text())
    hints = json.loads((folder / "repair_hints.json").read_text())
    samples = []
    groups = []
    execution_cache = {}
    for row in rows:
        task_samples = []
        schema = manifest["databases"][row["db_id"]]["schema"]
        pool = candidates(row["issue_sql"], hints[row["id"]]["suggestions"],
                          row["question"], schema, len(row["example"]["columns"]),
                          legacy=True)
        gold = references["gold_sql"][row["id"]]
        pool = list(dict.fromkeys([gold] + pool))[:80]
        cases = references["tasks"][row["id"]]
        for index, sql in enumerate(pool):
            outcomes = []
            for case in cases:
                identity = (case["path"], sql)
                if identity not in execution_cache:
                    result = local_sql(folder / case["path"], sql)
                    fingerprint = None
                    if result is not None:
                        try:
                            fingerprint = result_fingerprint(result, row["ordered"])
                        except Exception:
                            fingerprint = None
                    execution_cache[identity] = fingerprint
                outcomes.append(execution_cache[identity] == case["expected"])
            public_match = bool(outcomes[0])
            correct = all(outcomes)
            features = vector(sql, row["question"], schema,
                              len(row["example"]["columns"]), row["issue_sql"], index)
            weight = 0.5 if sql == gold else 1.0
            if correct:
                weight *= 3.0
            sample = (sql, features, {"public": public_match, "correct": correct}, weight)
            samples.append(sample)
            task_samples.append(sample)
        groups.append([sample for sample in task_samples if sample[2]["public"]])
    priority = fit(samples, "correct")
    matched = [sample for sample in samples if sample[2]["public"]]
    choice = fit(matched, "correct")
    choice_pair = fit_pairwise(groups)
    document = {
        "provenance": "trained on mp3_sql/data/train_references.json and repair_hints.json only",
        "split": "train",
        "samples": len(samples),
        "public_matches": len(matched),
        "correct_samples": sum(sample[2]["correct"] for sample in samples),
        "priority": priority,
        "choice": choice,
        "choice_pair": choice_pair,
    }
    output = ROOT / "submission" / "ranker.json"
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in document.items()
                      if key not in ("priority", "choice", "choice_pair")}))
    print("wrote", output)


if __name__ == "__main__":
    main()
