"""Development-only exact rollout simulator: real RepairAgent, synthetic tool feedback."""
import collections
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mp3_sql.common import load_bank, result_fingerprint  # noqa: E402
from train_ranker import local_sql  # noqa: E402
import submission as agent_module  # noqa: E402

OUT = ROOT / "outputs"
DATA = ROOT / "mp3_sql" / "data"


class Bank:
    def __init__(self, split):
        self.rows, self.manifest, self.folder = load_bank(split, DATA)
        self.references = json.loads((self.folder / f"{split}_references.json").read_text())
        self.by_id = {row["id"]: row for row in self.rows}
        self.cache = {}
        self.columns = {}

    def public_path(self, item):
        return self.manifest["databases"][item["db_id"]]["public_path"]

    def fingerprint(self, sql, path, ordered):
        key = (path, sql, ordered)
        if key not in self.cache:
            result = local_sql(self.folder / path, sql)
            self.cache[key] = result_fingerprint(result, ordered) if result else None
        return self.cache[key]

    def output_columns(self, item, sql):
        path = self.public_path(item)
        key = (path, sql)
        if key not in self.columns:
            result = local_sql(self.folder / path, sql)
            self.columns[key] = result["columns"] if result else []
        return self.columns[key]

    def public_match(self, item, sql):
        pool = item["pool"]
        if sql in pool:
            return item["public"][pool.index(sql)]
        expected = self.references["tasks"][item["id"]][0]["expected"]
        return self.fingerprint(sql, self.public_path(item), item["ordered"]) == expected

    def correct(self, item, sql):
        pool = item["pool"]
        if sql in pool:
            return item["correct"][pool.index(sql)]
        cases = self.references["tasks"][item["id"]]
        return all(self.fingerprint(sql, case["path"], item["ordered"]) == case["expected"]
                   for case in cases)


def rollout(item, bank, factory, schema_cached=False, log=None):
    agent = factory()
    remaining = 8
    history = []
    used = set()
    for _ in range(9):
        observation = {
            "task": {"id": item["id"], "question": item["question"]},
            "issue_sql": item["issue_sql"], "example": copy.deepcopy(item["example"]),
            "ordered": item["ordered"], "remaining": remaining,
            "history": copy.deepcopy(history),
        }
        action = agent.act(observation)
        tool = action["tool"]
        if tool == "submit":
            return {"sql": action["sql"], "tool_calls": 8 - remaining,
                    "history": history}
        assert remaining > 0, "budget exceeded"
        identity = json.dumps(action, sort_keys=True)
        assert identity not in used, "duplicate action"
        used.add(identity)
        if tool == "check" or tool == "execute":
            sql = action["sql"]
            matches = bank.public_match(item, sql)
            feedback = {"ok": matches, "columns": bank.output_columns(item, sql),
                        "rows": item["example"]["rows"] if matches else [],
                        "row_count": len(item["example"]["rows"]) if matches else 0,
                        "truncated": False, "error": None}
            if tool == "check":
                feedback["matches_example"] = matches
        elif tool == "repair_hint":
            feedback = list(item["suggestions"])
        elif tool == "schema":
            feedback = item["schema"]
        elif tool == "retrieve":
            feedback = []
        else:
            raise ValueError("unknown tool " + tool)
        history.append({**copy.deepcopy(action), "observation": feedback})
        remaining -= 1
    raise AssertionError("agent did not submit")


def main():
    weights = None
    splits = sys.argv[1:] or ("validation", "test")
    for split in splits:
        items = json.loads((OUT / f"pool-labels-{split}.json").read_text())
        bank = Bank(split)
        stats = collections.Counter()
        failures = []
        for item in items:
            result = rollout(item, bank, agent_module.create_agent)
            passed = bank.correct(item, result["sql"])
            stats["tasks"] += 1
            stats["correct"] += passed
            stats["tool_calls"] += result["tool_calls"]
            stats["any_correct_in_pool"] += any(item["correct"])
            if not passed:
                failures.append({"id": item["id"], "question": item["question"],
                                 "submitted": result["sql"],
                                 "has_correct": any(item["correct"]),
                                 "tool_calls": result["tool_calls"]})
        n = stats["tasks"]
        print(split, "correct", stats["correct"], "accuracy", round(100 * stats["correct"] / n, 2),
              "ceiling", stats["any_correct_in_pool"],
              "mean_tool_calls", round(stats["tool_calls"] / n, 3))
        (OUT / f"sim-failures-{split}.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
