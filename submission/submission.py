"""CPU-only, execution-guided SQL repair agent.

Loop: verify the draft -> read cached hints -> obtain schema (reused across tasks
through a memory cache) -> check ranked repair candidates -> choose a final query
that matched the public example. Every decision uses real tool feedback.
"""
from functools import lru_cache
import json
import os
import re
import threading

import ranker
import sqlglot
from sql_repairs import (candidates, parse, schema_swaps, semantic_templates,
                         simplifications)
from sqlglot import exp

# A schema tool result is deterministic for a database. Tasks arrive in arbitrary
# database order, so remember results from earlier tasks in this process and
# identify the current database from the draft's own table/column references.
_SCHEMA_CACHE = []
_SCHEMA_LOCK = threading.Lock()

DEFAULTS = {
    "priority_model": "priority2",
    "choice_model": "choice2",
    "feature_set": "vector2",
    "stop_matches": 1,
    "stop_matches_empty": 4,
    "schema_cache": True,
    "queue_mode": "model",
    "max_checks": 8,
    "immediate_submit": False,
    "schema_columns_required": 0,
    "choice_bonus": 0.3,
    "column_bonus": 2.0,
    "priority_column_bonus": 0.5,
    "choice_static_bonus": 0.0,
    "pin_first": True,
    "use_hints": True,
    "dedupe_queue": True,
    "dedupe_matches": True,
    "distinct_penalty": 0.0,
    "join_penalty": 0.0,
    "semijoin_bonus": 0.0,
    "swap_cap": 0,
    "swap_boost": 0.0,
    "choice_swap_bonus": 0.0,
    "wide_candidates": True,
    "pool_cap": 80,
    "distinct_relative_penalty": 3.0,
    "like_stopword_penalty": 0.0,
    "priority_negation_boost": 0.0,
    "priority_semijoin_boost": 0.0,
    "like_relative_penalty": 3.0,
    "semijoin_relative_bonus": 0.0,
    "original_relative_penalty": 0.0,
    "pk_count_bonus": 0.0,
    "sibling_exclude_original": False,
    "min_checks": 2,
    "priority_join_penalty": 0.0,
    "no_match_stop": 0,
}
ENV_KNOBS = {
    "priority_model": str, "choice_model": str, "feature_set": str,
    "stop_matches": int, "stop_matches_empty": int, "schema_cache": int,
    "queue_mode": str, "max_checks": int, "immediate_submit": int,
    "schema_columns_required": int, "choice_bonus": float,
    "column_bonus": float, "priority_column_bonus": float,
    "choice_static_bonus": float, "pin_first": int, "use_hints": int,
    "dedupe_queue": int, "dedupe_matches": int, "distinct_penalty": float,
    "join_penalty": float, "semijoin_bonus": float, "swap_cap": int,
    "swap_boost": float, "choice_swap_bonus": float,
    "wide_candidates": int, "pool_cap": int,
    "distinct_relative_penalty": float, "like_stopword_penalty": float,
    "priority_negation_boost": float, "priority_semijoin_boost": float,
    "like_relative_penalty": float, "semijoin_relative_bonus": float,
    "original_relative_penalty": float, "pk_count_bonus": float,
    "sibling_exclude_original": int, "min_checks": int,
    "priority_join_penalty": float, "no_match_stop": int,
}


@lru_cache(maxsize=20000)
def _canonical(sql):
    """Formatting-insensitive identity of a query; string literals stay caseful."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        return tree.sql(dialect="sqlite") if tree is not None else sql.strip()
    except Exception:
        return sql.strip()


LIKE_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "and", "or",
                  "with", "to", "is", "are", "was", "were"}


@lru_cache(maxsize=20000)
def _has_like(sql):
    tree = parse(sql)
    return bool(tree and any(isinstance(node, exp.Like) for node in tree.walk()))


@lru_cache(maxsize=20000)
def _like_stopwords(sql):
    """Verbatim question phrases copied into LIKE patterns often retain articles."""
    tree = parse(sql)
    if tree is None:
        return 0
    total = 0
    for node in tree.find_all(exp.Like):
        literal = node.args.get("expression")
        if isinstance(literal, exp.Literal) and literal.is_string:
            total += sum(word in LIKE_STOPWORDS
                         for word in re.findall(r"[a-z]+", literal.name.lower()))
    return total


def _pk_count(sql, schema):
    """Does COUNT(DISTINCT T.c) count the primary key of T?"""
    tree = parse(sql)
    if tree is None:
        return False
    maps = {table["name"].lower(): {column["name"].lower(): column.get("pk")
                                    for column in table["columns"]}
            for table in schema}
    aliases = {}
    for table in tree.find_all(exp.Table):
        aliases[(table.alias or table.name).lower()] = table.name.lower()
    for node in tree.find_all(exp.Count):
        if not node.args.get("distinct"):
            continue
        column = node.args.get("this")
        if not isinstance(column, exp.Column):
            continue
        owner = aliases.get(column.table.lower()) if column.table else None
        if owner and maps.get(owner, {}).get(column.name.lower()):
            return True
    return False


@lru_cache(maxsize=20000)
def _shape(sql):
    """Structural priors: DISTINCT, JOIN count, and pure semi-join form."""
    tree = parse(sql)
    if tree is None:
        return False, 0, False
    nodes = list(tree.walk())
    joins = sum(isinstance(node, exp.Join) for node in nodes)
    distinct = bool(tree.args.get("distinct"))
    semijoin = (not joins and any(
        isinstance(node, exp.In) and isinstance(node.args.get("query"), exp.Subquery)
        for node in nodes))
    return distinct, joins, semijoin


def _normalise_name(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _column_match(actual, expected):
    """Agreement between a checked query's output columns and the public example."""
    if not actual or not expected or len(actual) != len(expected):
        return -1.0
    left = [_normalise_name(name) for name in actual]
    right = [_normalise_name(name) for name in expected]
    exact = sum(a == b for a, b in zip(left, right)) / len(right)
    contained = sum(bool(a and b and (a in b or b in a))
                    for a, b in zip(left, right)) / len(right)
    return 0.6 * exact + 0.4 * contained


def _static_column_match(sql, expected):
    """Predict output column labels from the AST without executing the query."""
    tree = parse(sql)
    if tree is None or not expected:
        return -1.0
    names = [item.alias or item.sql(dialect="sqlite") for item in tree.expressions]
    return _column_match(names, expected)


def _simplicity(sql):
    """Occam prior: fewer joins/tables/subqueries is usually more robust."""
    tree = parse(sql)
    if tree is None:
        return 0.0
    nodes = list(tree.walk())
    joins = sum(isinstance(node, exp.Join) for node in nodes)
    tables = sum(isinstance(node, exp.Table) for node in nodes)
    subqueries = sum(isinstance(node, exp.Subquery) for node in nodes)
    distinct = 1.0 if tree.args.get("distinct") else 0.0
    projections = min(len(tree.expressions), 6)
    length = len(sql) / 200.0
    return -(0.8 * joins + 0.35 * max(tables - 1, 0) + 0.5 * subqueries +
             0.45 * distinct + 0.25 * length + 0.10 * projections)


def _identifiers(sql):
    tree = parse(sql)
    tables, columns = set(), set()
    if tree is not None:
        for node in tree.walk():
            if isinstance(node, exp.Table):
                tables.add(node.name.lower())
            elif isinstance(node, exp.Column):
                columns.add(node.name.lower())
    return tables, columns


def _remember_schema(schema):
    with _SCHEMA_LOCK:
        if schema not in _SCHEMA_CACHE:
            _SCHEMA_CACHE.append(schema)


def _infer_schema(sql, columns_required=0):
    """Only reuse a schema when the draft's identifiers identify it unambiguously."""
    tables, columns = _identifiers(sql)
    if not tables:
        return None
    with _SCHEMA_LOCK:
        cached = list(_SCHEMA_CACHE)
    scored = []
    for schema in cached:
        names = {table["name"].lower() for table in schema}
        if not tables <= names:
            continue
        available = set()
        for table in schema:
            available.update(column["name"].lower() for column in table["columns"])
        overlap = len(columns & available)
        if columns_required and overlap < columns_required:
            continue
        scored.append((overlap, schema))
    if not scored:
        return None
    best = max(score for score, _ in scored)
    winners = [schema for score, schema in scored if score == best]
    signature = json.dumps(winners[0], sort_keys=True)
    if any(json.dumps(schema, sort_keys=True) != signature for schema in winners):
        return None
    return winners[0]


class RepairAgent:
    def __init__(self, checkpoint=None):
        self.weights = ranker.load(checkpoint)
        self.config = dict(DEFAULTS)
        if self.weights is not None:
            self.config.update(self.weights.get("config", {}))
        for key, cast in ENV_KNOBS.items():
            value = os.environ.get("MP3_" + key.upper())
            if value is not None:
                self.config[key] = cast(value)
        self.queue = None
        self.pool = []
        self.context = None
        self.schema = None
        self.first_hint = None
        self.expected_columns = []
        self.example_rows = 0
        self.fallback = None
        self.matches = []
        self.checks = 0
        self.swaps = set()

    def _features(self, sql, index):
        question, schema, column_count, original = self.context
        if self.config["feature_set"] == "vector2":
            return ranker.vector2(sql, question, schema, column_count,
                                  original, index, self.first_hint, self.example_rows)
        return ranker.vector(sql, question, schema, column_count, original, index)

    def _priority(self, item):
        index, sql = item
        if self.weights is None:
            return -index / 1000.0
        value = ranker.raw(self.weights[self.config["priority_model"]],
                           self._features(sql, index))
        if self.config["priority_column_bonus"]:
            value += self.config["priority_column_bonus"] * _static_column_match(
                sql, self.expected_columns)
        if sql in self.swaps:
            value += self.config["swap_boost"]
        if self.config["priority_semijoin_boost"] and _shape(sql)[2]:
            value += self.config["priority_semijoin_boost"]
        if (self.config["priority_negation_boost"]
                and re.search(r"\b(no|not|never|without)\b", self.context[0].lower())
                and (" NOT IN " in sql.upper() or " EXCEPT " in sql.upper())):
            value += self.config["priority_negation_boost"]
        if self.config["priority_join_penalty"]:
            value -= self.config["priority_join_penalty"] * _shape(sql)[1]
        return value

    def _choose(self, observation):
        checked = {event["sql"]: event["observation"]
                   for event in observation["history"] if event["tool"] == "check"}
        matching = [sql for sql in self.matches if checked.get(sql, {}).get("matches_example")]
        # The original draft is known to be defective, so it should not act as a
        # trustworthy "simpler sibling" when contrastive priors are computed.
        if self.config["sibling_exclude_original"]:
            original_key = _canonical(observation["issue_sql"])
            peers = [sql for sql in matching if _canonical(sql) != original_key]
        else:
            peers = matching
        has_plain_sibling = any(not _shape(sql)[0] for sql in peers)
        has_plain_like = any(_has_like(sql) and _like_stopwords(sql) == 0
                             for sql in peers)
        has_join_sibling = any(_shape(sql)[1] for sql in peers)
        pk_counts = sum(_pk_count(sql, self.context[1]) for sql in matching)
        if matching:
            def score(sql):
                position = self.pool.index(sql) + 1 if sql in self.pool else 1
                if self.weights is None:
                    return -position
                value = ranker.raw(self.weights[self.config["choice_model"]],
                                   self._features(sql, position))
                value += self.config["choice_bonus"] * _simplicity(sql)
                if self.config["choice_static_bonus"]:
                    value += self.config["choice_static_bonus"] * _static_column_match(
                        sql, observation["example"]["columns"])
                if self.config["column_bonus"]:
                    agreement = _column_match(
                        checked[sql].get("columns"),
                        observation["example"]["columns"])
                    value += self.config["column_bonus"] * agreement
                distinct, joins, semijoin = _shape(sql)
                if distinct and not re.search(
                        r"\b(distinct|different|unique)\b", self.context[0].lower()):
                    value -= self.config["distinct_penalty"]
                value -= self.config["join_penalty"] * joins
                if semijoin:
                    value += self.config["semijoin_bonus"]
                if sql in self.swaps:
                    value += self.config["choice_swap_bonus"]
                if distinct and has_plain_sibling and not re.search(
                        r"\b(distinct|different|unique)\b", self.context[0].lower()):
                    value -= self.config["distinct_relative_penalty"]
                value -= self.config["like_stopword_penalty"] * _like_stopwords(sql)
                if (self.config["like_relative_penalty"] and has_plain_like
                        and _like_stopwords(sql)):
                    value -= self.config["like_relative_penalty"]
                if semijoin and has_join_sibling:
                    value += self.config["semijoin_relative_bonus"]
                if (len(matching) > 1 and sql == observation["issue_sql"]):
                    value -= self.config["original_relative_penalty"]
                if pk_counts >= 2 and _pk_count(sql, self.context[1]):
                    value += self.config["pk_count_bonus"]
                return value
            return {"tool": "submit", "sql": max(matching, key=score)}
        if checked.get(observation["issue_sql"], {}).get("matches_example"):
            return {"tool": "submit", "sql": observation["issue_sql"]}
        return {"tool": "submit", "sql": self.fallback or observation["issue_sql"]}

    def _build_queue(self, observation, schema, suggestions):
        original = observation["issue_sql"]
        question = observation["task"]["question"]
        column_count = len(observation["example"]["columns"])
        self.example_rows = len(observation["example"]["rows"])
        self.expected_columns = observation["example"]["columns"]
        self.first_hint = suggestions[0] if suggestions else original
        self.fallback = self.first_hint
        self.context = (question, schema, column_count, original)
        self.pool = candidates(
            original, suggestions, question, schema, column_count,
            wide=bool(self.config["wide_candidates"]),
            expected_columns=observation["example"]["columns"])[:self.config["pool_cap"]]
        self.swaps = set()
        if self.config["swap_cap"]:
            generated = []
            bases = list(dict.fromkeys(
                [self.first_hint] + suggestions[:2] + [original]))[:3]
            for base in bases:
                generated.extend(schema_swaps(base, question, schema,
                                              cap=self.config["swap_cap"]))
            swap_list = [sql for sql in dict.fromkeys(generated)
                         if sql not in self.pool][:self.config["swap_cap"]]
            self.swaps = set(swap_list)
            self.pool = self.pool + swap_list
        if not self.pool:
            self.queue = []
            return
        first = self.pool[0]
        scored = list(enumerate(self.pool, start=1))
        if self.config["queue_mode"] == "model":
            if self.config["pin_first"]:
                self.queue = [first] + [sql for _, sql in sorted(
                    (item for item in scored if item[1] != first),
                    key=self._priority, reverse=True)]
            else:
                self.queue = [sql for _, sql in sorted(scored, key=self._priority,
                                                       reverse=True)]
        else:
            ranked = sorted((item for item in scored if item[1] != first),
                            key=self._priority, reverse=True)
            semantic = []
            for source in (first, original):
                semantic.extend(semantic_templates(source, question, schema, column_count))
            preferred = [sql for sql in semantic if sql in self.pool]
            preferred += [sql for sql in simplifications(first, question) if sql in self.pool]
            seen = {first}
            self.queue = [first]
            for sql in preferred + [sql for _, sql in ranked]:
                if sql not in seen:
                    seen.add(sql)
                    self.queue.append(sql)
        if self.config["dedupe_queue"]:
            seen_keys, filtered = set(), []
            for sql in self.queue:
                key = _canonical(sql)
                if key not in seen_keys:
                    seen_keys.add(key)
                    filtered.append(sql)
            self.queue = filtered

    def act(self, observation):
        history = observation["history"]
        original = observation["issue_sql"]

        if not history:
            return {"tool": "check", "sql": original}

        tools = [event["tool"] for event in history]
        if "repair_hint" not in tools:
            return {"tool": "repair_hint"}

        if self.schema is None:
            if "schema" in tools:
                self.schema = next(event for event in history
                                   if event["tool"] == "schema")["observation"]
                _remember_schema(self.schema)
            elif self.config["schema_cache"]:
                self.schema = _infer_schema(original,
                                            self.config["schema_columns_required"])
            if self.schema is None:
                return {"tool": "schema"}

        checked = {event["sql"]: event["observation"]
                   for event in history if event["tool"] == "check"}
        if self.config["dedupe_queue"]:
            checked_keys = {_canonical(sql) for sql in checked}
        else:
            checked_keys = set(checked)
        self.matches = [sql for sql, result in checked.items()
                        if result.get("matches_example")]
        if self.config["dedupe_matches"]:
            seen_matches, unique_matches = set(), []
            for sql in self.matches:
                key = _canonical(sql)
                if key not in seen_matches:
                    seen_matches.add(key)
                    unique_matches.append(sql)
            match_count = len(unique_matches)
        else:
            match_count = len(self.matches)

        if self.queue is None:
            hint_event = next(event for event in history if event["tool"] == "repair_hint")
            suggestions = (list(dict.fromkeys(hint_event["observation"]))
                           if self.config["use_hints"] else [])
            self._build_queue(observation, self.schema, suggestions)

        empty_example = not observation["example"]["rows"]
        stop_matches = (self.config["stop_matches_empty"] if empty_example
                        else self.config["stop_matches"])
        last = history[-1]
        if (self.config["immediate_submit"] and last["tool"] == "check"
                and last["sql"] != original and last["observation"].get("matches_example")):
            return {"tool": "submit", "sql": last["sql"]}
        # If neither the draft nor any repair matched the public example, more
        # probing cannot produce a feedback-supported choice; stop early.
        no_match_stop = self.config["no_match_stop"]
        if (not observation["remaining"] or not self.queue
                or (match_count >= stop_matches
                    and self.checks >= self.config["min_checks"])
                or (no_match_stop and match_count == 0
                    and self.checks >= no_match_stop)
                or self.checks >= self.config["max_checks"]):
            return self._choose(observation)

        while self.queue:
            sql = self.queue.pop(0)
            if sql in checked or (self.config["dedupe_queue"]
                                  and _canonical(sql) in checked_keys):
                continue
            self.checks += 1
            return {"tool": "check", "sql": sql}
        return self._choose(observation)


def create_agent(checkpoint=None):
    return RepairAgent(checkpoint)
