"""Development-only: measure how well public example column names identify gold SQL."""
import collections
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mp3_sql.common import load_bank  # noqa: E402
from train_ranker import local_sql  # noqa: E402

OUT = ROOT / "outputs"
DATA = ROOT / "mp3_sql" / "data"


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def columns_match(actual, expected):
    if not actual or not expected or len(actual) != len(expected):
        return -1.0
    a = [norm(x) for x in actual]
    e = [norm(x) for x in expected]
    exact = sum(x == y for x, y in zip(a, e)) / len(e)
    contained = sum(bool(x and y and (x in y or y in x)) for x, y in zip(a, e)) / len(e)
    return 0.6 * exact + 0.4 * contained


def main(split):
    items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
    rows, manifest, folder = load_bank(split, DATA)
    by_id = {row["id"]: row for row in rows}
    cache = {}
    buckets = collections.defaultdict(lambda: [0, 0])
    improved = 0
    for item in items:
        row = by_id[item["id"]]
        public_path = manifest["databases"][row["db_id"]]["public_path"]
        expected = item["example"]["columns"]
        for sql, public, correct in zip(item["pool"], item["public"], item["correct"]):
            if not public:
                continue
            key = (public_path, sql)
            if key not in cache:
                result = local_sql(folder / public_path, sql)
                cache[key] = columns_match(result["columns"], expected) if result else -1.0
            score = cache[key]
            bucket = "high" if score >= 0.8 else ("mid" if score >= 0.3 else "low")
            pair = buckets[(bucket, "correct" if correct else "wrong")]
            pair[0] += 1
            pair[1] += 1
    print("split", split)
    for bucket in ("high", "mid", "low"):
        good = buckets[(bucket, "correct")][0]
        bad = buckets[(bucket, "wrong")][0]
        total = good + bad
        print(f"  {bucket:5s} n={total:4d} correct={good:4d} rate={good / max(total, 1):.3f}")
    # Oracle: if we always picked the highest column score among public matches
    groups = collections.defaultdict(list)
    for item in items:
        row = by_id[item["id"]]
        public_path = manifest["databases"][row["db_id"]]["public_path"]
        for sql, public, correct in zip(item["pool"], item["public"], item["correct"]):
            if public:
                groups[item["id"]].append((cache[(public_path, sql)], correct))
    picked = sum(max(group, key=lambda pair: pair[0])[1] for group in groups.values())
    best = sum(any(correct for _, correct in group) for group in groups.values())
    print("  column-oracle", picked, "ceiling", best, "of", len(items))


if __name__ == "__main__":
    for split in sys.argv[1:] or ("train", "validation", "test"):
        main(split)
