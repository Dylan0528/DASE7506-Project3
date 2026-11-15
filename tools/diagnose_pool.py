"""Development-only diagnostic: measure candidate-pool oracle and selection loss."""
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mp3_sql.common import load_bank, result_fingerprint  # noqa: E402
from sql_repairs import candidates  # noqa: E402
from ranker import load, predict, vector  # noqa: E402
from train_ranker import local_sql  # noqa: E402

DATA = ROOT / "mp3_sql" / "data"


def build_queue(row, schema, column_count, hints, weights):
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
    from sql_repairs import semantic_templates, simplifications
    semantic = []
    for source in (first, row["issue_sql"]):
        semantic.extend(semantic_templates(source, row["question"], schema, column_count))
    preferred = [sql for sql in semantic if sql in pool]
    preferred += [sql for sql in simplifications(first, row["question"]) if sql in pool]
    seen = {first}
    queue = [first]
    for sql in preferred + [sql for _, sql in ranked]:
        if sql not in seen:
            seen.add(sql)
            queue.append(sql)
    return pool, queue


def main(split):
    rows, manifest, folder = load_bank(split, DATA)
    references = json.loads((folder / f"{split}_references.json").read_text())
    hints = json.loads((folder / "repair_hints.json").read_text())
    weights = load()
    official_path = ROOT / "outputs" / ("ranked-v7-test-score.json" if split == "test"
                                        else "ranked-v7-validation-score.json")
    official = {}
    if official_path.is_file():
        official = {item["id"]: item["passed"] for item in
                    json.loads(official_path.read_text())["per_question"]}
    submissions = {}
    pred_path = ROOT / "outputs" / ("ranked-v7-test.json" if split == "test"
                                    else "ranked-v7-validation.json")
    if pred_path.is_file():
        doc = json.loads(pred_path.read_text())
        submissions = {item["id"]: item["sql"] for item in doc["predictions"]}

    cache = {}

    def fingerprint(sql, case):
        key = (case["path"], sql)
        if key not in cache:
            result = local_sql(folder / case["path"], sql)
            cache[key] = result_fingerprint(result, row["ordered"]) if result else None
        return cache[key]

    stats = collections.Counter()
    failures = []
    for row in rows:
        schema = manifest["databases"][row["db_id"]]["schema"]
        column_count = len(row["example"]["columns"])
        cases = references["tasks"][row["id"]]
        pool, queue = build_queue(row, schema, column_count, hints, weights)
        pub = {}
        for sql in pool:
            pub[sql] = fingerprint(sql, cases[0]) == cases[0]["expected"]
        matched = [sql for sql in pool if pub[sql]]
        correct = []
        for sql in matched:
            if all(fingerprint(sql, case) == case["expected"] for case in cases[1:]):
                correct.append(sql)
        stats["tasks"] += 1
        stats["has_public"] += bool(matched)
        stats["has_correct_in_pool"] += bool(correct)
        stats["first_queue_correct"] += bool(queue[0] in correct)
        stats["official_correct"] += bool(official.get(row["id"], False))
        stats["public_matches"] += len(matched)
        stats["correct_in_pool"] += len(correct)
        checked = [row["issue_sql"]] + [sql for sql in queue if sql != row["issue_sql"]][:5]
        stats["checked_has_correct"] += bool(set(checked) & set(correct))
        stats["checked_has_public"] += bool(set(checked) & set(matched))
        if official.get(row["id"], False) is False and correct:
            stats["official_fail_but_pool_correct"] += 1
            failures.append({
                "id": row["id"], "question": row["question"],
                "issue_sql": row["issue_sql"],
                "example": row["example"],
                "submitted": submissions.get(row["id"]),
                "n_matched": len(matched), "n_correct": len(correct),
                "correct": correct[:4],
                "checked": [sql for sql in checked if sql in matched][:6],
                "queue_head": queue[:8],
            })
    print(split, json.dumps(dict(stats), indent=2))
    out = ROOT / "outputs" / f"diagnose-{split}.json"
    out.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out, "failures", len(failures))


if __name__ == "__main__":
    for split in sys.argv[1:] or ("validation", "test"):
        main(split)
