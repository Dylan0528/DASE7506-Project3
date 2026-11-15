"""Development-only: precompute public-match / 3-instance labels for candidate pools."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mp3_sql.common import load_bank, result_fingerprint  # noqa: E402
from sql_repairs import candidates  # noqa: E402
from train_ranker import local_sql  # noqa: E402

DATA = ROOT / "mp3_sql" / "data"
OUT = ROOT / "outputs"


def build(split, wide=False):
    rows, manifest, folder = load_bank(split, DATA)
    references = json.loads((folder / f"{split}_references.json").read_text())
    hints = json.loads((folder / "repair_hints.json").read_text())
    cache = {}

    def fingerprint(sql, path, ordered):
        key = (path, sql)
        if key not in cache:
            result = local_sql(folder / path, sql)
            cache[key] = result_fingerprint(result, ordered) if result else None
        return cache[key]

    data = []
    for row in rows:
        schema = manifest["databases"][row["db_id"]]["schema"]
        column_count = len(row["example"]["columns"])
        pool = candidates(row["issue_sql"], hints[row["id"]]["suggestions"],
                          row["question"], schema, column_count, wide=wide,
                          expected_columns=row["example"]["columns"])[:80]
        cases = references["tasks"][row["id"]]
        public, correct = [], []
        for sql in pool:
            match = fingerprint(sql, cases[0]["path"], row["ordered"]) == cases[0]["expected"]
            public.append(match)
            passed = False
            if match:
                passed = all(fingerprint(sql, case["path"], row["ordered"]) == case["expected"]
                             for case in cases[1:])
            correct.append(passed)
        data.append({
            "id": row["id"], "db_id": row["db_id"], "question": row["question"],
            "issue_sql": row["issue_sql"], "example": row["example"],
            "ordered": row["ordered"], "column_count": column_count,
            "schema": schema, "pool": pool, "public": public, "correct": correct,
            "suggestions": list(dict.fromkeys(hints[row["id"]]["suggestions"])),
        })
    suffix = "-wide" if wide else ""
    path = OUT / f"pool-labels-{split}{suffix}.json"
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    print(split, "wide", wide, "tasks", len(data),
          "with correct", sum(any(item["correct"]) for item in data),
          "candidates", sum(len(item["pool"]) for item in data),
          "public matches", sum(sum(item["public"]) for item in data),
          "correct", sum(sum(item["correct"]) for item in data), "->", path)


if __name__ == "__main__":
    wide = "--wide" in sys.argv
    splits = [arg for arg in sys.argv[1:] if not arg.startswith("--")] or \
        ("train", "validation", "test")
    for split in splits:
        build(split, wide)
