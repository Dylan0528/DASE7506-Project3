"""Development-only grid search over final-choice blends on released splits."""
import collections
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))

from mp3_sql.common import load_bank, result_fingerprint  # noqa: E402
from ranker import load, predict, vector  # noqa: E402
from sql_repairs import (candidates, priority_templates, semantic_templates,  # noqa: E402
                         simplifications)
from train_ranker import local_sql  # noqa: E402


def build_queue(row, schema, column_count, hints, weights, original_empty=False):
    pool = candidates(row["issue_sql"], hints[row["id"]]["suggestions"],
                      row["question"], schema, column_count)[:80]
    first = pool[0]
    def priority(item):
        index, sql = item
        features = vector(sql, row["question"], schema, column_count,
                          row["issue_sql"], index)
        return predict(weights["priority"], features)
    ranked = sorted(((index, sql) for index, sql in enumerate(pool, start=1)
                     if sql != first), key=priority, reverse=True)
    preferred = []
    boosted = []
    if os.environ.get("SEM_MODE") == "highprec":
        boosted = priority_templates(row["issue_sql"], first, row["question"],
                                     schema, column_count, original_empty)
        preferred = list(boosted)
    semantic_count = int(os.environ.get("SEM_N", "0")) if not boosted else 0
    if semantic_count:
        semantic = []
        for source in (first, row["issue_sql"]):
            semantic.extend(semantic_templates(source, row["question"],
                                               schema, column_count))
        unique = [sql for sql in dict.fromkeys(semantic) if sql in pool]
        if os.environ.get("SEM_MODE") == "order":
            preferred.extend(unique[:semantic_count])
        else:
            scored = []
            for sql in unique:
                position = pool.index(sql) + 1
                scored.append((priority((position, sql)), sql))
            preferred.extend(sql for _, sql in sorted(scored, reverse=True)[:semantic_count])
    preferred += [sql for sql in simplifications(first, row["question"]) if sql in pool]
    seen = {first}
    queue = [first]
    for sql in preferred + [sql for _, sql in ranked]:
        if sql not in seen:
            seen.add(sql)
            queue.append(sql)
    return pool, queue, set(boosted)


def main(split):
    rows, manifest, folder = load_bank(split)
    references = json.loads((folder / f"{split}_references.json").read_text())
    hints = json.loads((folder / "repair_hints.json").read_text())
    weights = load()
    boost = float(os.environ.get("SEM_BOOST", "0"))
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    tally = collections.defaultdict(int)
    cache = {}
    for row in rows:
        schema = manifest["databases"][row["db_id"]]["schema"]
        column_count = len(row["example"]["columns"])
        cases = references["tasks"][row["id"]]
        original_result = local_sql(folder / cases[0]["path"], row["issue_sql"])
        original_empty = bool(original_result and not original_result["rows"])
        pool, queue, boosted = build_queue(row, schema, column_count, hints,
                                           weights, original_empty)
        checked = [row["issue_sql"]] + [sql for sql in queue
                                      if sql != row["issue_sql"]][:5]

        def fingerprint(sql, case):
            key = (case["path"], sql)
            if key not in cache:
                result = local_sql(folder / case["path"], sql)
                cache[key] = result_fingerprint(result, row["ordered"]) if result else None
            return cache[key]

        matching = [sql for sql in checked if fingerprint(sql, cases[0]) == cases[0]["expected"]]
        correct = {sql for sql in checked
                   if all(fingerprint(sql, case) == case["expected"] for case in cases)}
        fallback_correct = queue[0] in correct
        scores = {}
        for sql in matching:
            position = pool.index(sql) + 1 if sql in pool else 1
            features = vector(sql, row["question"], schema, column_count,
                              row["issue_sql"], position)
            scores[sql] = (predict(weights["choice"], features),
                           predict(weights["choice_pair"], features))
        for k in range(1, 6):
            visible = set(checked[:k + 1]) & set(matching)
            for alpha in alphas:
                if visible:
                    chosen = max(visible, key=lambda sql:
                                 alpha * scores[sql][0] + (1 - alpha) * scores[sql][1] +
                                 (boost if sql in boosted else 0.0))
                    passed = chosen in correct
                else:
                    passed = fallback_correct
                tally[(k, alpha)] += passed
    best = sorted(tally.items(), key=lambda item: item[1], reverse=True)[:10]
    print(split, "n", len(rows), "best", best)
    for key in sorted(tally):
        print("K", key[0], "alpha", key[1], "correct", tally[key],
              "accuracy", round(100 * tally[key] / len(rows), 2))


if __name__ == "__main__":
    for split in sys.argv[1:] or ("validation",):
        main(split)
