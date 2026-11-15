"""Development-only: estimate how much a schema-guided edit space could add.

For tasks where the current candidate pool contains no query that passes all
three grading instances, generate simple one-edit rewrites (column swaps,
literal/pattern changes, negation and DISTINCT toggles) from the public hints
and label them offline against the released references.  This is a *diagnostic*
only: the shipped agent never reads references or grading databases.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mp3_sql.common import load_bank, result_fingerprint  # noqa: E402
from sql_repairs import parse, render  # noqa: E402
from sqlglot import exp  # noqa: E402
from train_ranker import local_sql  # noqa: E402

OUT = ROOT / "outputs"
DATA = ROOT / "mp3_sql" / "data"
_CACHE = {}


def fingerprint(sql, path, ordered, folder):
    key = (path, sql, ordered)
    if key not in _CACHE:
        result = local_sql(folder / path, sql)
        _CACHE[key] = result_fingerprint(result, ordered) if result else None
    return _CACHE[key]


def correct(sql, cases, ordered, folder):
    if fingerprint(sql, cases[0]["path"], ordered, folder) != cases[0]["expected"]:
        return False
    return all(fingerprint(sql, case["path"], ordered, folder) == case["expected"]
               for case in cases[1:])


def schema_columns(schema):
    result = {}
    for table in schema:
        result[table["name"].lower()] = [column["name"] for column in table["columns"]]
    return result


def variants(sql, schema, cap=400):
    tree = parse(sql)
    if tree is None:
        return []
    maps = schema_columns(schema)
    aliases = {}
    for table in tree.find_all(exp.Table):
        if table.name.lower() in maps:
            aliases[(table.alias or table.name).lower()] = table.name.lower()
    output = []

    def emit(candidate):
        if len(output) >= cap:
            return
        try:
            text = render(candidate)
        except Exception:
            return
        if text not in output:
            output.append(text)

    for node in list(tree.find_all(exp.Column)):
        qualifier = node.table.lower()
        owner = aliases.get(qualifier)
        pool = maps.get(owner, []) if owner else \
            [name for names in maps.values() for name in names]
        for name in pool[:14]:
            if name.lower() == node.name.lower():
                continue
            candidate = deepcopy(tree)
            for target in list(candidate.find_all(exp.Column)):
                if target.sql(dialect="sqlite") == node.sql(dialect="sqlite"):
                    target.set("this", exp.to_identifier(name))
            emit(candidate)

    for node in list(tree.find_all(exp.Like)):
        literal = node.args.get("expression")
        if not isinstance(literal, exp.Literal) or not literal.is_string:
            continue
        value = literal.name.strip("%")
        for pattern in ("%" + value + "%", value + "%", "%" + value):
            if pattern == literal.name:
                continue
            candidate = deepcopy(tree)
            for target in list(candidate.find_all(exp.Like)):
                if target.sql(dialect="sqlite") == node.sql(dialect="sqlite"):
                    target.set("expression", exp.Literal.string(pattern))
            emit(candidate)

    for node in list(tree.find_all(exp.EQ)):
        literal = node.args.get("expression")
        if not isinstance(literal, exp.Literal) or not literal.is_string:
            continue
        candidate = deepcopy(tree)
        for target in list(candidate.find_all(exp.EQ)):
            if target.sql(dialect="sqlite") == node.sql(dialect="sqlite"):
                target.replace(exp.Like(this=deepcopy(target.this),
                                        expression=exp.Literal.string("%" + literal.name + "%")))
        emit(candidate)

    for flag in (True, False):
        candidate = deepcopy(tree)
        candidate.set("distinct", exp.Distinct() if flag else None)
        emit(candidate)
    for node in list(tree.find_all(exp.In)):
        for negated in (True, False):
            candidate = deepcopy(tree)
            for target in list(candidate.find_all(exp.In)):
                if target.sql(dialect="sqlite") == node.sql(dialect="sqlite"):
                    target.set("not", negated or None)
            emit(candidate)
    return output


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    rows, manifest, folder = load_bank(split, DATA)
    references = json.loads((folder / f"{split}_references.json").read_text())
    labels = json.loads((OUT / f"pool-labels-{split}.json").read_text())
    failures = json.loads((OUT / f"analyze-{split}.json").read_text())
    missing = {item["id"] for item in failures if item["kind"] == "no_candidate"}
    by_id = {item["id"]: item for item in labels}
    row_by_id = {row["id"]: row for row in rows}
    wins, details = 0, []
    for identifier in sorted(missing):
        item = by_id[identifier]
        row = row_by_id[identifier]
        cases = references["tasks"][identifier]
        schema = item["schema"]
        generated, found = [], None
        for base in list(dict.fromkeys(item["suggestions"] + [item["issue_sql"]]))[:3]:
            for text in variants(base, schema):
                if text in item["pool"]:
                    continue
                generated.append(text)
                if correct(text, cases, item["ordered"], folder):
                    found = (base, text)
                    break
            if found:
                break
        if found:
            wins += 1
            details.append({"id": identifier, "question": item["question"],
                            "base": found[0], "winner": found[1],
                            "generated": len(generated)})
    print(split, "no-candidate tasks", len(missing), "reachable by one edit", wins,
          "generated", sum(item["generated"] for item in details))
    for detail in details:
        print("---", detail["id"], detail["question"][:110])
        print(" base:", detail["base"].replace("\n", " ")[:220])
        print(" win :", detail["winner"].replace("\n", " ")[:220])
    (OUT / f"probe-edit-space-{split}.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
