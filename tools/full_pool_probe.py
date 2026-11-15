"""Development-only: is the 80-candidate cap losing correct repairs?"""
import collections
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mp3_sql.common import load_bank, result_fingerprint  # noqa: E402
from sql_repairs import candidates, parse  # noqa: E402
from train_ranker import local_sql  # noqa: E402

OUT = ROOT / "outputs"
DATA = ROOT / "mp3_sql" / "data"


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def agreement(sql, expected):
    tree = parse(sql)
    if tree is None or not expected:
        return -1.0
    names = [item.alias or item.sql(dialect="sqlite") for item in tree.expressions]
    if len(names) != len(expected):
        return -1.0
    a = [norm(x) for x in names]
    e = [norm(x) for x in expected]
    exact = sum(x == y for x, y in zip(a, e)) / len(e)
    contained = sum(bool(x and y and (x in y or y in x)) for x, y in zip(a, e)) / len(e)
    return 0.6 * exact + 0.4 * contained


def main(split):
    rows, manifest, folder = load_bank(split, DATA)
    refs = json.loads((folder / f"{split}_references.json").read_text())
    hints = json.loads((folder / "repair_hints.json").read_text())
    cache = {}

    def fp(sql, path, ordered):
        key = (path, sql)
        if key not in cache:
            result = local_sql(folder / path, sql)
            cache[key] = result_fingerprint(result, ordered) if result else None
        return cache[key]

    stats = collections.Counter()
    for row in rows:
        schema = manifest["databases"][row["db_id"]]["schema"]
        cc = len(row["example"]["columns"])
        full = candidates(row["issue_sql"], hints[row["id"]]["suggestions"],
                          row["question"], schema, cc)
        cases = refs["tasks"][row["id"]]
        stats["tasks"] += 1
        stats["full_size_total"] += len(full)
        stats["over_80"] += len(full) > 80
        labels = []
        for pos, sql in enumerate(full):
            matched = fp(sql, cases[0]["path"], row["ordered"]) == cases[0]["expected"]
            correct = matched and all(fp(sql, case["path"], row["ordered"]) == case["expected"]
                                      for case in cases[1:])
            labels.append((pos, sql, correct))
        stats["correct_full"] += any(correct for _, _, correct in labels)
        stats["correct_80"] += any(correct for pos, _, correct in labels if pos < 80)
        scored = sorted(labels, key=lambda item: -agreement(item[1], row["example"]["columns"]))
        stats["correct_top80_by_columns"] += any(item[2] for item in scored[:80])
        stats["correct_top40_by_columns"] += any(item[2] for item in scored[:40])
    print(split, json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    for split in sys.argv[1:] or ("validation", "test"):
        main(split)
