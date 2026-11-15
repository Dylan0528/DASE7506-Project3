"""Development-only: inspect which SQL traits separate 3-instance-correct candidates."""
import collections
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "tools"))

from sql_repairs import parse  # noqa: E402
from sqlglot import exp  # noqa: E402

OUT = ROOT / "outputs"


def traits(sql):
    tree = parse(sql)
    lower = sql.lower()
    t = {
        "like": " like " in lower,
        "eq_literal": bool(re.search(r"=\s*'", lower)),
        "not_in": " not in " in lower,
        "in_select": " in (select" in lower,
        "except": " except " in lower,
        "union": " union " in lower,
        "distinct": "distinct" in lower,
        "limit": " limit " in lower,
        "order": " order by " in lower,
        "group": " group by " in lower,
        "having": " having " in lower,
        "agg": bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", lower)),
        "subquery": bool(re.search(r"\(\s*select", lower)),
        "star": "*" in lower,
        "cast": " cast(" in lower,
        "length": min(len(sql), 600) / 600,
    }
    if tree is not None:
        nodes = list(tree.walk())
        t["joins"] = min(len(tree.args.get("joins") or []), 5)
        t["tables"] = min(sum(isinstance(n, exp.Table) for n in nodes), 6)
        t["projections"] = min(len(tree.expressions), 6)
    else:
        t.update({"joins": 0, "tables": 0, "projections": 0})
    return t


def main(split="train"):
    items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
    sums = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
    for item in items:
        for sql, public, correct in zip(item["pool"], item["public"], item["correct"]):
            tr = traits(sql)
            for key, value in tr.items():
                bucket = "pub" if public else "no_pub"
                target = sums[(bucket, key)]
                target[0] += float(value)
                target[1] += 1
                if public:
                    pair = sums[("correct" if correct else "wrong", key)]
                    pair[2] += float(value)
                    pair[3] += 1
    keys = sorted({key for _, key in sums})
    print(f"{'trait':14s} {'public':>8s} {'correct':>8s} {'wrong':>8s} {'lift':>8s}")
    for key in keys:
        pub = sums[("pub", key)]
        cor = sums[("correct", key)]
        bad = sums[("wrong", key)]
        p = pub[0] / max(pub[1], 1)
        c = cor[2] / max(cor[3], 1)
        w = bad[2] / max(bad[3], 1)
        lift = (c + 0.01) / (w + 0.01)
        print(f"{key:14s} {p:8.3f} {c:8.3f} {w:8.3f} {lift:8.2f}")


if __name__ == "__main__":
    for split in sys.argv[1:] or ("train",):
        print("===", split)
        main(split)
