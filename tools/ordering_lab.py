"""Development-only: evaluate candidate queue orderings against released labels."""
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ranker import load, predict, vector  # noqa: E402
from sql_repairs import semantic_templates, simplifications  # noqa: E402

OUT = ROOT / "outputs"


def load_split(split):
    return json.loads((OUT / f"pool-labels-{split}.json").read_text())


def prepare(data, weights):
    prepared = []
    for item in data:
        vectors = []
        for index, sql in enumerate(item["pool"], start=1):
            vectors.append(vector(sql, item["question"], item["schema"],
                                  item["column_count"], item["issue_sql"], index))
        prepared.append((item, vectors))
    return prepared


def current_queue(item, vectors, weights):
    pool = item["pool"]
    first = pool[0]
    ranked = sorted(((index, sql) for index, sql in enumerate(pool, start=1)
                     if sql != first),
                    key=lambda pair: predict(weights["priority"], vectors[pair[0] - 1]),
                    reverse=True)
    semantic = []
    for source in (first, item["issue_sql"]):
        semantic.extend(semantic_templates(source, item["question"],
                                           item["schema"], item["column_count"]))
    preferred = [sql for sql in semantic if sql in pool]
    preferred += [sql for sql in simplifications(first, item["question"]) if sql in pool]
    seen = {first}
    queue = [first]
    for sql in preferred + [sql for _, sql in ranked]:
        if sql not in seen:
            seen.add(sql)
            queue.append(sql)
    return queue


def by_model(name):
    def order(item, vectors, weights):
        return sorted(range(len(item["pool"])),
                      key=lambda i: predict(weights[name], vectors[i]), reverse=True)
    return lambda item, vectors, weights: [item["pool"][i] for i in order(item, vectors, weights)]


def simulate(prepared, weights, queue_fn, k=5):
    stats = collections.Counter()
    for item, vectors in prepared:
        pool = item["pool"]
        public = dict(zip(pool, item["public"]))
        correct = dict(zip(pool, item["correct"]))
        queue = queue_fn(item, vectors, weights)
        queue = list(queue) + [sql for sql in pool if sql not in set(queue)]
        checked = [item["issue_sql"]] + [sql for sql in queue
                                         if sql != item["issue_sql"]][:k]
        matching = [sql for sql in checked if public.get(sql, False)]
        stats["tasks"] += 1
        stats["has_public"] += bool(any(item["public"]))
        stats["has_correct"] += bool(any(item["correct"]))
        stats["checked_has_correct"] += bool(any(correct.get(sql, False) for sql in checked))
        if matching:
            chosen = max(matching, key=lambda sql: predict(
                weights["choice"], vectors[pool.index(sql)]))
            passed = correct.get(chosen, False)
        else:
            passed = correct.get(queue[0], False)
        stats["correct"] += passed
    return stats


def main():
    weights = load()
    splits = sys.argv[1:] or ("validation", "test")
    strategies = {
        "current": current_queue,
        "priority": by_model("priority"),
        "choice": by_model("choice"),
        "choice_pair": by_model("choice_pair"),
    }
    for split in splits:
        print("===", split)
        prepared = prepare(load_split(split), weights)
        for name, fn in strategies.items():
            for k in (4, 5):
                stats = simulate(prepared, weights, fn, k)
                print(f"{name:14s} K={k} correct={stats['correct']:3d} "
                      f"acc={100 * stats['correct'] / stats['tasks']:6.2f} "
                      f"ceiling={stats['has_correct']:3d} "
                      f"checked_correct={stats['checked_has_correct']:3d}")


if __name__ == "__main__":
    main()
