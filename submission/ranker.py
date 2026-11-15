"""Compact, CPU-only feature ranker trained on the released training split."""
import json
import math
from copy import deepcopy
from pathlib import Path
import re

from sql_repairs import parse
from sqlglot import exp

STOP = set("a an the of for in on at to and or with all show me find list what "
           "which who whose whom when where how many are is was were do does did "
           "give return please their its it that this those these each per "
           "there have has had be been by from as than then also".split())


def tokens(text):
    return set(re.findall(r"[a-z][a-z0-9_]*", text.lower())) - STOP


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-min(value, 35)))
    return math.exp(max(value, -35)) / (1 + math.exp(max(value, -35)))


def _flag(condition):
    return 1.0 if condition else 0.0


def vector(sql, question, schema, column_count, original=None, index=0):
    words = tokens(question)
    lower_q = question.lower()
    lower_s = sql.lower()
    tree = parse(sql)
    columns, tables, literals, projections = set(), set(), set(), set()
    if tree is not None:
        for node in tree.walk():
            if isinstance(node, exp.Column):
                columns.add(node.name.lower())
            elif isinstance(node, exp.Table):
                tables.add(node.name.lower())
            elif isinstance(node, exp.Literal):
                literals.add(node.name.lower())
        for item in tree.expressions:
            for node in item.walk():
                if isinstance(node, exp.Column):
                    projections.add(node.name.lower())
    identifiers = columns | tables
    schema_names = {column["name"].lower() for table in schema for column in table["columns"]}
    schema_names |= {table["name"].lower() for table in schema}

    q = {
        "count": _flag(re.search(r"\b(how many|number of|count)\b", lower_q)),
        "distinct": _flag(re.search(r"\b(distinct|different|unique)\b", lower_q)),
        "total": _flag(re.search(r"\b(total|sum)\b", lower_q)),
        "average": _flag("average" in lower_q or " avg " in lower_q),
        "maximum": _flag(re.search(r"\b(maximum|max|most|largest|highest|oldest|youngest)\b", lower_q)),
        "minimum": _flag(re.search(r"\b(minimum|min|least|smallest|lowest)\b", lower_q)),
        "group": _flag(re.search(r"\b(each|per|for every)\b", lower_q)),
        "comparison": _flag(re.search(r"\b(at least|at most|more than|less than|above|below)\b", lower_q)),
        "negation": _flag(re.search(r"\b(no|not|never|without|exclude|do not|does not)\b", lower_q)),
        "both": _flag("both" in lower_q or re.search(r"\b(and)\b.*\b(and)\b", lower_q)),
        "order": _flag(re.search(r"\b(ascending|descending|alphabetical|ordered|order by)\b", lower_q)),
        "limit": _flag(re.search(r"\b(top|first|last|limit)\b", lower_q)),
        "name": _flag("name" in words or "names" in lower_q),
        "id": _flag("id" in words or "ids" in lower_q),
    }
    if tree is None:
        structure = {"parse": 0.0}
        counts = {}
    else:
        all_nodes = list(tree.walk())
        structure = {
            "parse": 1.0,
            "projections": min(len(tree.expressions), 6),
            "distinct": _flag(tree.args.get("distinct")),
            "joins": min(len(tree.args.get("joins") or []), 6),
            "tables": min(sum(isinstance(n, exp.Table) for n in all_nodes), 8),
            "subqueries": min(sum(isinstance(n, exp.Subquery) for n in all_nodes), 5),
            "aggregates": min(sum(isinstance(n, exp.AggFunc) for n in all_nodes), 6),
            "group": _flag(tree.args.get("group")),
            "having": _flag(tree.args.get("having")),
            "order": _flag(tree.args.get("order")),
            "limit": _flag(tree.args.get("limit")),
            "where": _flag(tree.args.get("where")),
            "not": _flag(any(isinstance(n, (exp.Not, exp.NEQ)) for n in all_nodes)),
            "in": _flag(any(isinstance(n, exp.In) for n in all_nodes)),
            "exists": _flag(any(isinstance(n, exp.Exists) for n in all_nodes)),
            "union": _flag(any(isinstance(n, (exp.Union, exp.Except, exp.Intersect)) for n in all_nodes)),
            "like": _flag(any(isinstance(n, exp.Like) for n in all_nodes)),
            "cast": _flag(any(isinstance(n, exp.Cast) for n in all_nodes)),
            "case": _flag(any(isinstance(n, exp.Case) for n in all_nodes)),
            "count": _flag(any(isinstance(n, exp.Count) for n in all_nodes)),
            "sum": _flag(any(isinstance(n, exp.Sum) for n in all_nodes)),
            "avg": _flag(any(isinstance(n, exp.Avg) for n in all_nodes)),
            "minmax": _flag(any(isinstance(n, (exp.Min, exp.Max)) for n in all_nodes)),
            "not_in": _flag("not in" in lower_s),
            "exists_not": _flag("not exists" in lower_s),
            "left_join": _flag("left join" in lower_s),
            "count_star": _flag("count(*)" in lower_s.replace(" ", "")),
            "count_distinct": _flag("count(distinct" in lower_s.replace(" ", "")),
        }
        counts = {
            "predicates": min(sum(isinstance(n, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ))
                                 for n in all_nodes), 8),
            "boolean": min(sum(isinstance(n, (exp.And, exp.Or)) for n in all_nodes), 6),
            "sql_length": min(len(sql) / 200, 5),
            "column_delta": min(abs(len(tree.expressions) - column_count), 4),
        }

    coverage = {
        "identifier_overlap": len(words & identifiers) / max(len(words), 1),
        "projection_overlap": len(words & projections) / max(len(words), 1),
        "table_overlap": len(words & tables) / max(len(words), 1),
        "literal_overlap": len(words & literals) / max(len(words), 1),
        "schema_identifiers": sum(name in schema_names for name in identifiers) / max(len(identifiers), 1),
        "unknown_identifiers": sum(name not in schema_names for name in identifiers) / max(len(identifiers), 1),
        "question_length": min(len(words) / 20, 2),
    }
    cross = {
        "count_agg": q["count"] * structure.get("count", 0),
        "count_without_agg": q["count"] * (1 - structure.get("aggregates", 0) / 6),
        "total_sum": q["total"] * structure.get("sum", 0),
        "average_avg": q["average"] * structure.get("avg", 0),
        "maximum_minmax": q["maximum"] * structure.get("minmax", 0),
        "maximum_order": q["maximum"] * structure.get("order", 0),
        "minimum_minmax": q["minimum"] * structure.get("minmax", 0),
        "group_group": q["group"] * structure.get("group", 0),
        "group_without": q["group"] * (1 - structure.get("group", 0)),
        "distinct_distinct": q["distinct"] * structure.get("distinct", 0),
        "distinct_absent": (1 - q["distinct"]) * structure.get("distinct", 0),
        "negation_not": q["negation"] * structure.get("not", 0),
        "negation_absent": q["negation"] * (1 - structure.get("not", 0)),
        "both_and": q["both"] * counts.get("boolean", 0),
        "comparison_having": q["comparison"] * structure.get("having", 0),
        "order_order": q["order"] * structure.get("order", 0),
        "limit_limit": q["limit"] * structure.get("limit", 0),
        "name_projection": q["name"] * _flag("name" in projections),
        "id_projection": q["id"] * _flag("id" in projections or any(c.endswith("_id") for c in projections)),
    }
    origin = {
        "is_original": _flag(original is not None and sql == original),
        "first_hint": _flag(index == 1),
        "rank": min(index / 20, 1),
        "rank_log": math.log1p(index) / 3,
    }
    result = {}
    for group in (q, structure, counts, coverage, cross, origin):
        result.update({key: float(value) for key, value in group.items()})
    return result


def raw(weights, features):
    """Unsquashed linear score; monotone in predict but better for blending."""
    means = weights["means"]
    scales = weights["scales"]
    coefficients = weights["coefficients"]
    keys = weights["keys"]
    value = weights["intercept"]
    for key in keys:
        z = (features.get(key, 0.0) - means[key]) / scales[key]
        value += z * coefficients[key]
    return max(min(value, 30), -30)


def predict(weights, features):
    return sigmoid(raw(weights, features))


def _core(tree):
    """SQL of a query with its projection list removed, for structural diffs."""
    if tree is None:
        return ""
    copy = deepcopy(tree)
    copy.set("expressions", [])
    try:
        return copy.sql(dialect="sqlite")
    except Exception:
        return ""


def _string_literals(tree, node_type):
    result = set()
    if tree is None:
        return result
    for node in tree.walk():
        if isinstance(node, node_type) and isinstance(node.expression, exp.Literal):
            if node.expression.is_string:
                result.add(node.expression.name.lower())
    return result


def vector2(sql, question, schema, column_count, original=None, index=0,
            first_hint=None, example_rows=0):
    """Add schema-independent robustness and repair-diff features."""
    result = vector(sql, question, schema, column_count, original, index)
    tree = parse(sql)
    orig = parse(original) if original else None
    lower = sql.lower()
    orig_lower = (original or "").lower()
    nodes = list(tree.walk()) if tree is not None else []
    orig_nodes = list(orig.walk()) if orig is not None else []

    def count(items, kind):
        return sum(isinstance(node, kind) for node in items)

    extra = {
        "kw_in_select": _flag(" in (select" in lower),
        "kw_not_in": _flag(" not in " in lower),
        "kw_except": _flag(" except " in lower),
        "kw_like_literal": _flag(bool(re.search(r"\blike\s+'", lower))),
        "kw_eq_literal": _flag(bool(re.search(r"=\s*'", lower))),
        "kw_star_projection": _flag(any(isinstance(item.this, exp.Star)
                                         for item in (tree.expressions if tree else []))),
        "kw_order": _flag(" order by " in lower),
        "kw_limit": _flag(" limit " in lower),
        "kw_group": _flag(" group by " in lower),
        "kw_having": _flag(" having " in lower),
        "n_eq_literal": min(len(re.findall(r"=\s*'", lower)), 4),
        "n_like_literal": min(len(re.findall(r"\blike\s+'", lower)), 4),
        "n_in_select": min(lower.count(" in (select"), 3),
        "example_empty": _flag(example_rows == 0),
        "example_rows": min(example_rows / 10.0, 2),
        "arity_ok": _flag(tree is not None and len(tree.expressions) == column_count),
        "same_as_original": _flag(original is not None and sql == original),
        "same_as_first_hint": _flag(first_hint is not None and sql == first_hint),
        "is_first_hint": _flag(index == 1),
    }
    extra["fragility"] = (extra["n_in_select"] + extra["kw_not_in"] +
                          count(nodes, exp.Subquery) / 3 +
                          count(nodes, exp.Join) / 3 + extra["kw_eq_literal"])
    if tree is not None and orig is not None:
        extra["diff_projection_only"] = _flag(
            _core(tree) == _core(orig) and
            [item.sql() for item in tree.expressions] !=
            [item.sql() for item in orig.expressions])
        extra["diff_eq_to_like"] = _flag(bool(
            _string_literals(orig, exp.EQ) & _string_literals(tree, exp.Like)))
        extra["diff_like_pattern"] = _flag(
            _string_literals(orig, exp.Like) != _string_literals(tree, exp.Like) and
            bool(_string_literals(tree, exp.Like)))
        extra["diff_join_removed"] = min(max(count(orig_nodes, exp.Join) -
                                             count(nodes, exp.Join), 0), 3)
        extra["diff_table_removed"] = min(max(count(orig_nodes, exp.Table) -
                                              count(nodes, exp.Table), 0), 3)
        extra["diff_subquery_removed"] = min(max(count(orig_nodes, exp.Subquery) -
                                                 count(nodes, exp.Subquery), 0), 3)
        extra["diff_shorter"] = min(max(len(original) - len(sql), 0) / 200.0, 2)
        extra["diff_distinct_removed"] = _flag(
            orig.args.get("distinct") and not tree.args.get("distinct"))
        extra["diff_where_removed"] = _flag(
            orig.args.get("where") and not tree.args.get("where"))
        extra["diff_limit_added"] = _flag(
            not orig.args.get("limit") and tree.args.get("limit"))
        extra["diff_group_added"] = _flag(
            not orig.args.get("group") and tree.args.get("group"))
        extra["diff_except_added"] = _flag(" except " in lower and " except " not in orig_lower)
        extra["diff_in_added"] = _flag(" in (select" in lower and " in (select" not in orig_lower)
    else:
        for key in ("diff_projection_only", "diff_eq_to_like", "diff_like_pattern",
                    "diff_join_removed", "diff_table_removed", "diff_subquery_removed",
                    "diff_shorter", "diff_distinct_removed", "diff_where_removed",
                    "diff_limit_added", "diff_group_added", "diff_except_added",
                    "diff_in_added"):
            extra[key] = 0.0
    result.update({key: float(value) for key, value in extra.items()})
    return result


def load(path=None):
    if path is None:
        path = Path(__file__).with_name("ranker.json")
    else:
        path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())
