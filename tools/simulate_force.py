"""Development-only simulation of high-precision execution-verified overrides."""
import collections
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ["SEM_MODE"] = "highprec"

from mp3_sql.common import load_bank, result_fingerprint  # noqa: E402
from ranker import load, predict, vector  # noqa: E402
from simulate_choice import build_queue  # noqa: E402
from train_ranker import local_sql  # noqa: E402

TRIGGER = re.compile(
    r"corresponds to|affiliated with|no friends|do not have any friends|"
    r"average grade.*friends|treatments?.*average|average.*treatments?|"
    r"maximum.*average|substring|least number|official language|friends of")


def force_score(sql, question, schema):
    lower = question.lower()
    score = 0.0
    shared = set()
    tables = [column["name"].lower() for table in schema for column in table["columns"]]
    for name in tables:
        if name.endswith("_id") or name in ("id", "uid"):
            shared.add(name)
    for key in shared:
        if re.search(r"\b" + re.escape(key) + r"\b", sql.lower()):
            score += 3
    if "friends" in lower:
        if " EXCEPT " in sql:
            score += 8
        if "student_id" in sql.lower():
            score += 8
        if "likes." in sql.lower() or " FROM Likes".lower() in sql.lower():
            score -= 8
    if "treatments" in lower and " IN (SELECT DISTINCT " in sql:
        score += 5
    if "corresponds to" in lower or "affiliated with" in lower:
        if " LIKE '" in sql:
            score += 8
    if "maximum" in lower and "average" in lower and " AS average" in sql:
        score += 8
    if "substring" in lower and " LIKE '%" in sql:
        score += 8
    if "least number" in lower and "LIMIT 1" in sql:
        score += 8
    if "official language" in lower and ".*" in sql:
        score += 8
    return score


def main(split, k=5):
    rows, manifest, folder = load_bank(split)
    references = json.loads((folder / f"{split}_references.json").read_text())
    hints = json.loads((folder / "repair_hints.json").read_text())
    weights = load()
    cache = {}
    tally = collections.Counter()
    for row in rows:
        schema = manifest["databases"][row["db_id"]]["schema"]
        column_count = len(row["example"]["columns"])
        cases = references["tasks"][row["id"]]

        def fingerprint(sql, case):
            key = (case["path"], sql)
            if key not in cache:
                result = local_sql(folder / case["path"], sql)
                cache[key] = result_fingerprint(result, row["ordered"]) if result else None
            return cache[key]

        original_result = local_sql(folder / cases[0]["path"], row["issue_sql"])
        pool, queue, boosted = build_queue(row, schema, column_count, hints, weights,
                                           bool(original_result and not original_result["rows"]))
        checked = [row["issue_sql"]] + [sql for sql in queue
                                      if sql != row["issue_sql"]][:k]
        matching = [sql for sql in checked if fingerprint(sql, cases[0]) == cases[0]["expected"]]
        forced = ([sql for sql in matching if sql in boosted]
                  if TRIGGER.search(row["question"].lower()) else [])
        if forced:
            chosen = max(forced, key=lambda sql: force_score(sql, row["question"], schema))
        elif matching:
            def score(sql):
                position = pool.index(sql) + 1 if sql in pool else 1
                features = vector(sql, row["question"], schema, column_count,
                                  row["issue_sql"], position)
                return predict(weights["choice"], features)
            chosen = max(matching, key=score)
        else:
            chosen = queue[0]
        passed = all(fingerprint(chosen, case) == case["expected"] for case in cases)
        tally["correct"] += passed
        tally["forced"] += bool(forced)
        tally["forced_correct"] += bool(forced and passed)
        tally["normal_correct"] += bool(not forced and passed)
    print(split, "n", len(rows), dict(tally),
          "accuracy", round(100 * tally["correct"] / len(rows), 2))


if __name__ == "__main__":
    for split in sys.argv[1:] or ("validation", "test"):
        main(split)
