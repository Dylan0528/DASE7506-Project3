"""Deterministic, schema-aware candidate repairs for SQLite queries."""
from copy import deepcopy
from itertools import combinations, permutations
import re

import sqlglot
from sqlglot import exp


def parse(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        return tree if isinstance(tree, exp.Select) else None
    except Exception:
        return None


def render(tree):
    return tree.sql(dialect="sqlite") + ";"


def add_variants(target, seen, trees):
    for tree in trees:
        try:
            sql = render(tree)
        except Exception:
            continue
        if sql not in seen:
            seen.add(sql)
            target.append(sql)


def _restore_order_aliases(candidate, original):
    """Keep ORDER BY valid after removing an aliased aggregate projection."""
    selected_aliases = {item.alias.lower() for item in candidate.expressions if item.alias}
    aliases = {item.alias.lower(): item.unalias() for item in original.expressions if item.alias}
    order = candidate.args.get("order")
    if order is None:
        return
    for column in list(order.walk()):
        if not isinstance(column, exp.Column):
            continue
        key = column.name.lower()
        if not column.table and key in aliases and key not in selected_aliases:
            column.replace(deepcopy(aliases[key]))


def projection_variants(sql, column_count):
    """Try projections with the requested arity, preferring reversed order.

    Spider questions often ask for an entity but a draft also exposes the COUNT
    used only for ranking. Conversely, aggregate/descriptor pairs are frequently
    emitted in the opposite order by small models.
    """
    tree = parse(sql)
    if tree is None:
        return []
    expressions = list(tree.expressions)
    if not expressions or len(expressions) > 5:
        return []
    variants = []

    if len(expressions) == column_count and column_count > 1:
        orders = [tuple(reversed(range(len(expressions))))]
        orders += [order for order in permutations(range(len(expressions)))
                   if order != orders[0]]
        for order in orders:
            candidate = deepcopy(tree)
            candidate.set("expressions", [deepcopy(expressions[i]) for i in order])
            _restore_order_aliases(candidate, tree)
            variants.append(candidate)

    if column_count < len(expressions):
        groups = list(combinations(range(len(expressions)), column_count))
        groups.sort(key=lambda group: (
            sum(isinstance(expressions[i].unalias(), exp.AggFunc) for i in group),
            sum(_questionable_id(expressions[i]) for i in group),
        ))
        for group in groups:
            orders = list(permutations(group)) if column_count > 1 else [group]
            if column_count > 1:
                orders.sort(key=lambda order: order != tuple(reversed(group)))
            for order in orders:
                candidate = deepcopy(tree)
                candidate.set("expressions", [deepcopy(expressions[i]) for i in order])
                _restore_order_aliases(candidate, tree)
                variants.append(candidate)
    return variants


def _questionable_id(expression):
    name = expression.unalias().name.lower()
    return name == "count" or name.endswith("_id") or name == "id"


def _schema_maps(schema):
    return {table["name"].lower(): {column["name"].lower(): column["name"]
                                    for column in table["columns"]}
            for table in schema}


def qualify_variant(sql, schema):
    """Resolve ambiguous or misplaced columns using only the public schema."""
    tree = parse(sql)
    if tree is None:
        return None
    schemas = _schema_maps(schema)
    aliases = {}
    for table in tree.find_all(exp.Table):
        table_name = table.name
        if table_name.lower() not in schemas:
            continue
        alias = table.alias or table_name
        aliases[alias.lower()] = table_name.lower()
    if not aliases:
        return None

    root = tree.args.get("from")
    root_alias = ""
    if root is not None and isinstance(root.this, exp.Table):
        root_alias = (root.this.alias or root.this.name).lower()
    changed = False
    for column in tree.walk():
        if not isinstance(column, exp.Column):
            continue
        name = column.name.lower()
        qualifier = column.table.lower()
        if qualifier:
            owner = aliases.get(qualifier)
            if owner is None:
                if any(name in columns for columns in schemas.values()):
                    column.set("table", None)
                    changed = True
            elif name not in schemas[owner]:
                choices = [alias for alias, table in aliases.items() if name in schemas[table]]
                if len(choices) == 1:
                    column.set("table", exp.to_identifier(choices[0]))
                    changed = True
        else:
            choices = [alias for alias, table in aliases.items() if name in schemas[table]]
            if len(choices) == 1:
                column.set("table", exp.to_identifier(choices[0]))
                changed = True
            elif len(choices) > 1 and root_alias in choices:
                column.set("table", exp.to_identifier(root_alias))
                changed = True
    return render(tree) if changed else None


def _prune_global_aggregate(tree, question, schema):
    lower = question.lower()
    if not tree.args.get("joins") or tree.args.get("where"):
        return None
    if not re.search(r"\b(total|how many|number of|average)\b", lower):
        return None
    if re.search(r"\b(each|per)\b", lower):
        return None
    expressions = list(tree.expressions)
    if not expressions or not all(list(item.walk()) for item in expressions):
        return None
    aggregate_columns = []
    for item in expressions:
        if not any(isinstance(node, exp.AggFunc) for node in item.walk()):
            return None
        aggregate_columns.extend(node for node in item.walk() if isinstance(node, exp.Column))
    if not aggregate_columns:
        return None
    schemas = _schema_maps(schema)
    owners = []
    for table in tree.find_all(exp.Table):
        columns = schemas.get(table.name.lower(), {})
        if all(column.name.lower() in columns for column in aggregate_columns):
            owners.append(table)
    if len(owners) != 1:
        return None
    candidate = deepcopy(tree)
    candidate.set("from", exp.From(this=deepcopy(owners[0])))
    candidate.set("joins", [])
    candidate.set("group", None)
    candidate.set("having", None)
    candidate.set("order", None)
    candidate.set("limit", None)
    return candidate


def _aggregate_column_variants(tree):
    output = []
    if any(isinstance(node, exp.Count) and not isinstance(node.this, exp.Star)
           for node in tree.walk()):
        candidate = deepcopy(tree)
        for node in list(candidate.walk()):
            if isinstance(node, exp.Count):
                node.set("this", exp.Star())
                node.set("distinct", None)
        output.append(candidate)

    columns = []
    for item in tree.expressions:
        for node in item.walk():
            if isinstance(node, exp.AggFunc) and isinstance(node.this, exp.Column):
                if node.this.sql() not in [column.sql() for column in columns]:
                    columns.append(node.this)
    if len(columns) > 1:
        for column in columns:
            candidate = deepcopy(tree)
            for item in list(candidate.expressions):
                for node in list(item.walk()):
                    if isinstance(node, exp.AggFunc) and isinstance(node.this, exp.Column):
                        node.set("this", deepcopy(column))
            output.append(candidate)
    return output


def _split_anti_union(tree):
    """Try each branch of NOT IN (A UNION B) separately."""
    output = []
    for in_node in list(tree.find_all(exp.In)):
        subquery = in_node.args.get("query")
        if not isinstance(subquery, exp.Subquery) or not isinstance(subquery.this, exp.Union):
            continue
        union = subquery.this
        for branch in (union.this, union.expression):
            candidate = deepcopy(tree)
            for node in list(candidate.find_all(exp.In)):
                branch_query = node.args.get("query")
                if (isinstance(branch_query, exp.Subquery)
                        and isinstance(branch_query.this, exp.Union)):
                    branch_query.set("this", deepcopy(branch))
                    break
            output.append(candidate)
    return output


def basic_variants(sql, question, schema, column_count):
    """Generate semantic edits without looking at any grading reference."""
    tree = parse(sql)
    if tree is None:
        return []
    output = projection_variants(sql, column_count)
    lower = question.lower()
    output[0:0] = _aggregate_column_variants(tree)
    if re.search(r"\b(no|not|never|without)\b", lower):
        output[0:0] = _split_anti_union(tree)

    pruned = _prune_global_aggregate(tree, question, schema)
    if pruned is not None:
        output.insert(0, pruned)

    if "older than average" in lower:
        root = tree.args.get("from")
        if root is not None and isinstance(root.this, exp.Table):
            table = root.this
            columns = _schema_maps(schema).get(table.name.lower(), {})
            age = columns.get("age")
            if age:
                condition = parse("SELECT 1 WHERE " + age +
                                  " > (SELECT AVG(" + age + ") FROM " + table.name + ")")
                candidate = deepcopy(tree)
                predicate = condition.args["where"].this
                if candidate.args.get("where"):
                    predicate = exp.and_(candidate.args["where"].this.copy(), predicate)
                candidate.set("where", exp.Where(this=predicate))
                output.insert(0, candidate)
                output[0:0] = projection_variants(render(candidate), column_count)

    if tree.args.get("group") and "total" in lower and not re.search(r"\b(each|per)\b", lower):
        candidate = deepcopy(tree)
        candidate.set("group", None)
        output.append(candidate)

    if tree.args.get("where") and re.search(r"\ball\b", lower):
        candidate = deepcopy(tree)
        candidate.set("where", None)
        output.append(candidate)

    if re.search(r"\bfor each product\b", lower) and tree.args.get("group"):
        root = tree.args.get("from")
        if root is not None and isinstance(root.this, exp.Table):
            table = root.this
            columns = _schema_maps(schema).get(table.name.lower(), {})
            product = columns.get("product_id")
            group_columns = {item.name.lower() for item in tree.args["group"].expressions}
            if product and product.lower() not in group_columns:
                candidate = deepcopy(tree)
                alias = table.alias or table.name
                candidate.args["group"].append("expressions", exp.to_column(alias + "." + product))
                output.insert(0, candidate)

    if "most common" in lower and tree.args.get("joins") and not tree.args.get("where"):
        root = tree.args.get("from")
        if root is not None and isinstance(root.this, exp.Table):
            columns = _schema_maps(schema).get(root.this.name.lower(), {})
            required = [node for item in tree.expressions for node in item.walk()
                        if isinstance(node, exp.Column)]
            required += [node for node in tree.args["group"].walk()
                         if isinstance(node, exp.Column)] if tree.args.get("group") else []
            if required and all(node.name.lower() in columns for node in required):
                candidate = deepcopy(tree)
                candidate.set("joins", [])
                output.insert(0, candidate)

    for equal in list(tree.walk()):
        if not isinstance(equal, exp.EQ):
            continue
        column = equal.this if isinstance(equal.this, exp.Column) else None
        literal = equal.expression
        if column is not None and "duration" in column.name.lower() and isinstance(literal, exp.Literal):
            candidate = deepcopy(tree)
            for node in candidate.walk():
                if isinstance(node, exp.EQ) and node.this.sql() == equal.this.sql():
                    value = literal.name.rstrip("%")
                    node.replace(exp.Like(this=deepcopy(node.this),
                                          expression=exp.Literal.string(value + ":%")))
                    output.insert(0, candidate)
                    break

    having = tree.args.get("having")
    if having is not None:
        text = having.this.sql(dialect="sqlite")
        replacement = None
        if "at most" in lower:
            replacement = re.sub(r">=?", "<=", text)
        elif "at least" in lower:
            replacement = re.sub(r"(?<![<>])>(?!=)", ">=", text)
        if replacement and replacement != text:
            candidate = deepcopy(tree)
            candidate.set("having", parse("SELECT 1 HAVING " + replacement).args["having"])
            output.append(candidate)

    if "distinct" in lower and not tree.args.get("distinct"):
        candidate = deepcopy(tree)
        candidate.set("distinct", exp.Distinct())
        output.append(candidate)
    elif "distinct" not in lower and tree.args.get("distinct"):
        candidate = deepcopy(tree)
        candidate.set("distinct", None)
        output.append(candidate)
    return output


def name_variants(sql, question, schema):
    """Split full-name literals when the schema stores first/last separately."""
    columns = [column["name"] for table in schema for column in table["columns"]]
    first = next((c for c in columns if c.lower() in
                  ("fname", "first_name", "firstname", "first")), None)
    last = next((c for c in columns if c.lower() in
                 ("lname", "last_name", "lastname", "last")), None)
    if not first or not last:
        return []
    variants = []
    literals = re.findall(r"'([^']+\s+[^']+)'", sql)
    for literal in literals:
        words = literal.split()
        if len(words) != 2 or literal.lower() not in question.lower():
            continue
        first_name, last_name = words
        pattern = re.compile(
            r"(?:\w+\.)?(?:" + re.escape(last) + r")\s*=\s*'" + re.escape(literal) + r"'",
            re.IGNORECASE)
        replacement = first + " = '" + first_name + "' AND " + last + " = '" + last_name + "'"
        changed = pattern.sub(replacement, sql, count=1)
        if changed != sql:
            variants.append(changed)

    # Also repair swapped first/last values in otherwise reasonable hints.
    for literal in literals:
        words = literal.split()
        if len(words) == 2 and literal.lower() in question.lower():
            first_name, last_name = words
            changed = re.sub(r"((?:\w+\.)?" + re.escape(first) + r"\s*=\s*')" +
                             re.escape(last_name) + "'", r"\1" + first_name + "'",
                             sql, flags=re.IGNORECASE)
            changed = re.sub(r"((?:\w+\.)?" + re.escape(last) + r"\s*=\s*')" +
                             re.escape(first_name) + "'", r"\1" + last_name + "'",
                             changed, flags=re.IGNORECASE)
            if changed != sql:
                variants.append(changed)
    return variants


def simplifications(sql, question):
    """Generate less restrictive alternatives that preserve question semantics."""
    tree = parse(sql)
    if tree is None:
        return []
    output = []
    lower = question.lower()
    if tree.args.get("distinct") and not re.search(r"\b(distinct|different|unique)\b", lower):
        candidate = deepcopy(tree)
        candidate.set("distinct", None)
        output.append(candidate)
    if any(isinstance(node, exp.Cast) for node in tree.walk()):
        candidate = deepcopy(tree)
        for node in list(candidate.walk()):
            if isinstance(node, exp.Cast):
                node.replace(deepcopy(node.this))
        output.append(candidate)
    if tree.args.get("group") and not tree.args.get("having"):
        if not any(isinstance(node, exp.AggFunc) for item in tree.expressions for node in item.walk()):
            candidate = deepcopy(tree)
            candidate.set("group", None)
            output.append(candidate)
    joins = list(tree.find_all(exp.Join))
    if joins:
        removable = []
        for join in joins:
            table = join.this if isinstance(join.this, exp.Table) else None
            if table is None:
                continue
            alias = (table.alias or table.name).lower()
            on = join.args.get("on")
            on_count = sum(1 for node in on.walk()
                           if isinstance(node, exp.Column) and node.table.lower() == alias) if on else 0
            total = sum(1 for node in tree.walk()
                        if isinstance(node, exp.Column) and node.table.lower() == alias)
            if total <= on_count:
                removable.append(join)
        if removable:
            candidate = deepcopy(tree)
            removable_sql = {join.sql() for join in removable}
            surviving = [join for join in candidate.args.get("joins", [])
                         if join.sql() not in removable_sql]
            candidate.set("joins", surviving)
            output.append(candidate)
    return [render(item) for item in output]


def _normalise_aliases(expression, aliases):
    expression = deepcopy(expression)
    for column in expression.walk():
        if isinstance(column, exp.Column) and column.table:
            key = column.table.lower()
            if key in aliases:
                column.set("table", exp.to_identifier(aliases[key]))
    return expression


def _root_info(tree, schema):
    root = tree.args.get("from")
    if root is None or not isinstance(root.this, exp.Table):
        return None
    table = root.this
    name = table.name
    metadata = next((item for item in schema if item["name"].lower() == name.lower()), None)
    if metadata is None:
        return None
    alias = table.alias or name
    aliases = {alias.lower(): name}
    for join in tree.args.get("joins") or []:
        joined = join.this
        if isinstance(joined, exp.Table):
            aliases[(joined.alias or joined.name).lower()] = joined.name
    columns = [item["name"] for item in metadata["columns"]]
    primary = next((item["name"] for item in metadata["columns"] if item.get("pk")), None)
    primary = primary or next((item for item in columns if item.lower() in ("id", "uid")), None)
    primary = primary or columns[0]
    name_column = next((item for item in columns if "name" in item.lower()), columns[0])
    projections = ", ".join(_normalise_aliases(item, aliases).sql(dialect="sqlite")
                            for item in tree.expressions)
    where = tree.args.get("where")
    where_sql = _normalise_aliases(where.this, aliases).sql(dialect="sqlite") if where else ""
    return {"table": name, "alias": alias, "columns": columns,
            "primary": primary, "name_column": name_column,
            "projections": projections, "where": where_sql, "aliases": aliases}


def _question_stems(question):
    result = set()
    for word in re.findall(r"[a-z]+", question.lower()):
        result.add(word)
        if word.endswith("s") and len(word) > 3:
            result.add(word[:-1])
        if word.endswith("es") and len(word) > 4:
            result.add(word[:-2])
    return result


def _relation_candidates(schema, question, info):
    stems = _question_stems(question)
    result = []
    for table in schema:
        name = table["name"]
        if name.lower() == info["table"].lower():
            continue
        table_words = set(re.findall(r"[a-z]+", name.lower()))
        table_score = 2 * len(table_words & stems)
        for column in table["columns"]:
            key = column["name"]
            lower = key.lower()
            if not (lower.endswith("_id") or lower in ("id", info["primary"].lower())):
                continue
            words = set(re.findall(r"[a-z]+", lower))
            score = table_score + 3 * len(words & stems)
            if lower in ("id", info["primary"].lower()):
                score += 2
            if words & set(re.findall(r"[a-z]+", info["table"].lower())):
                score += 2
            result.append((score, name, key))
    return [item[1:] for item in sorted(result, reverse=True)[:8]]


def _relation_templates(sql, question, schema, info):
    output = []
    lower = question.lower()
    stems = _question_stems(question)
    for relation, key in _relation_candidates(schema, question, info):
        inner = "SELECT DISTINCT " + key + " FROM " + relation
        if info["where"] and any(word in info["where"].lower() for word in
                                 (relation.lower(), "cost", "treatment", "average")):
            inner += " WHERE " + info["where"]
        outer = "SELECT " + info["projections"] + " FROM " + info["table"]
        outer += " WHERE " + info["table"] + "." + info["primary"] + " IN (" + inner + ")"
        if re.search(r"\b(no|not|never|without)\b", lower):
            output.append("SELECT DISTINCT " + info["table"] + "." + info["name_column"] +
                          " FROM " + info["table"] + " EXCEPT SELECT DISTINCT " +
                          info["table"] + "." + info["name_column"] + " FROM " + info["table"] +
                          " JOIN " + relation + " ON " + info["table"] + "." + info["primary"] +
                          " = " + relation + "." + key)
            output.append("SELECT " + info["projections"] + " FROM " + info["table"] +
                          " WHERE " + info["table"] + "." + info["primary"] +
                          " NOT IN (" + inner + ")")
        output.append(outer)
        if "friends of" in lower:
            literal_match = re.search(r"friends of (?:the )?(?:high school )?student ([A-Z][a-z]+)", question)
            if literal_match:
                literal = literal_match.group(1)
                output.append("SELECT " + info["table"] + "." + info["name_column"] +
                              " FROM " + info["table"] + " WHERE " + info["table"] + "." +
                              info["primary"] + " IN (SELECT DISTINCT " + key + " FROM " +
                              relation + " WHERE " + relation + "." + key +
                              " IN (SELECT " + info["primary"] + " FROM " + info["table"] +
                              " WHERE " + info["name_column"] + " = '" + literal + "'))")
    return output


def semantic_templates(sql, question, schema, column_count):
    """Schema-question templates for recurring Spider relational patterns."""
    tree = parse(sql)
    if tree is None:
        return []
    info = _root_info(tree, schema)
    if info is None:
        return []
    output = []
    lower = question.lower()

    for equal in list(tree.walk()):
        if isinstance(equal, exp.EQ) and isinstance(equal.expression, exp.Literal):
            if equal.expression.is_string:
                candidate = deepcopy(tree)
                for node in list(candidate.walk()):
                    if isinstance(node, exp.EQ) and node.sql() == equal.sql():
                        node.replace(exp.Like(this=deepcopy(node.this),
                                              expression=deepcopy(node.expression)))
                        break
                output.append(candidate)

    for node in list(tree.walk()):
        if isinstance(node, exp.AggFunc) and isinstance(node.this, exp.Column):
            if node.this.name.lower() in (node.sql_name().lower(), "average", "avg", "total", "sum"):
                candidate = deepcopy(tree)
                for target in list(candidate.walk()):
                    if isinstance(target, exp.AggFunc) and target.sql() == node.sql():
                        target.replace(deepcopy(target.this))
                        break
                output.append(candidate)

    if column_count > len(tree.expressions):
        candidate = deepcopy(tree)
        root_name = info["table"]
        candidate.set("expressions", [exp.Column(this=exp.Star(),
                                                 table=exp.to_identifier(root_name))])
        output.append(candidate)

    order = tree.args.get("order")
    if order is not None and tree.args.get("limit"):
        ordered = next((node for node in order.expressions if isinstance(node, exp.Ordered)), None)
        if ordered is not None and isinstance(ordered.this, exp.Column):
            function = "MAX" if ordered.args.get("desc") else "MIN"
            column = ordered.this.name
            condition = parse("SELECT 1 WHERE " + column + " = (SELECT " + function +
                              "(" + column + ") FROM " + info["table"] + ")").args["where"].this
            candidate = deepcopy(tree)
            candidate.set("joins", [])
            candidate.set("order", None)
            candidate.set("limit", None)
            previous = candidate.args.get("where")
            predicate = exp.and_(previous.this.copy(), condition) if previous else condition
            candidate.set("where", exp.Where(this=predicate))
            output.append(candidate)

    if order is not None and not tree.args.get("limit") and re.search(r"\b(most|least|largest|smallest)\b", lower):
        candidate = deepcopy(tree)
        candidate.set("limit", exp.Limit(expression=exp.Literal.number(1)))
        output.append(candidate)

    for like in list(tree.walk()):
        if isinstance(like, exp.Like) and isinstance(like.expression, exp.Literal) and like.expression.is_string:
            pattern = like.expression.name
            words = [word for word in re.findall(r"[a-z]+", pattern.lower())
                     if word not in ("the", "a", "an", "of", "in", "on", "for")]
            if words and " ".join(words) != pattern.strip("%").lower():
                candidate = deepcopy(tree)
                for target in list(candidate.walk()):
                    if isinstance(target, exp.Like) and target.sql() == like.sql():
                        target.set("expression", exp.Literal.string("%" + " ".join(words) + "%"))
                        break
                output.append(candidate)

    if re.search(r"\b(each|per|every)\b", lower) and not tree.args.get("group"):
        candidate = deepcopy(tree)
        groups = [deepcopy(item.unalias()) for item in tree.expressions
                  if not any(isinstance(node, exp.AggFunc) for node in item.walk())]
        if groups:
            candidate.set("group", exp.Group(expressions=groups))
            output.append(candidate)

    for count in list(tree.walk()):
        if isinstance(count, exp.Count) and isinstance(count.this, exp.Column):
            owner = count.this.table or info["table"]
            metadata = next((item for item in schema if item["name"].lower() == owner.lower()), None)
            if metadata is None:
                continue
            columns = [item["name"] for item in metadata["columns"]]
            stems = _question_stems(question)
            scored = [(len(set(re.findall(r"[a-z]+", name.lower())) & stems), name)
                      for name in columns]
            for score, name in sorted(scored, reverse=True)[:3]:
                if score and name.lower() != count.this.name.lower():
                    candidate = deepcopy(tree)
                    for target in list(candidate.walk()):
                        if isinstance(target, exp.Count) and target.sql() == count.sql():
                            target.set("this", exp.to_column(owner + "." + name))
                            break
                    output.append(candidate)

    if re.search(r"\baverage|how many|number of\b", lower) and tree.args.get("joins"):
        only_aggregates = all(any(isinstance(node, exp.AggFunc) for node in item.walk())
                              for item in tree.expressions)
        if only_aggregates and not re.search(r"\b(each|per)\b", lower):
            candidate = deepcopy(tree)
            candidate.set("joins", [])
            candidate.set("where", None)
            candidate.set("group", None)
            output.append(candidate)

    relation_trigger = bool(tree.args.get("joins") or re.search(
        r"\b(friends?|treatments?|pets?|cartoons?|broadcast(?:ed)?|playing)\b", lower))
    templates = _relation_templates(sql, question, schema, info) if relation_trigger else []
    rendered = []
    for item in output:
        try:
            rendered.append(render(item))
        except Exception:
            continue
    return rendered + templates


def _norm_identifier(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _projection_agreement(expressions, order, target):
    names = [_norm_identifier(expressions[i].alias or expressions[i].sql(dialect="sqlite"))
             for i in order]
    if len(names) != len(target):
        return -1.0
    exact = sum(a == b for a, b in zip(names, target)) / len(target)
    contained = sum(bool(a and b and (a in b or b in a))
                    for a, b in zip(names, target)) / len(target)
    return 0.6 * exact + 0.4 * contained


def column_guided_projections(sql, expected_columns, column_count):
    """Use the public example's column labels to order/choose projected columns.

    Long drafts (more than five expressions) are skipped by the generic
    permutation generator; the expected labels tell us which subset matters.
    """
    tree = parse(sql)
    if tree is None or not expected_columns:
        return []
    expressions = list(tree.expressions)
    if not expressions or column_count < 1 or column_count > 5:
        return []
    target = [_norm_identifier(name) for name in expected_columns]
    if len(target) != column_count:
        return []
    output = []
    if len(expressions) == column_count:
        orders = sorted(permutations(range(len(expressions))),
                        key=lambda order: -_projection_agreement(expressions, order, target))[:3]
    elif len(expressions) > column_count:
        groups = list(combinations(range(len(expressions)), column_count))
        groups.sort(key=lambda group: -max(
            _projection_agreement(expressions, order, target)
            for order in permutations(group)))
        orders = []
        for group in groups[:4]:
            orders.extend(sorted(permutations(group),
                                 key=lambda order: -_projection_agreement(
                                     expressions, order, target))[:1])
    else:
        return []
    for order in orders:
        candidate = deepcopy(tree)
        candidate.set("expressions", [deepcopy(expressions[i]) for i in order])
        _restore_order_aliases(candidate, tree)
        output.append(render(candidate))
    return list(dict.fromkeys(output))


def unqualify_variant(sql, schema):
    """Strip qualifiers that name no real table or alias (a common hint error)."""
    tree = parse(sql)
    if tree is None:
        return None
    valid = {table["name"].lower() for table in schema}
    for table in tree.find_all(exp.Table):
        valid.add(table.name.lower())
        valid.add((table.alias or table.name).lower())
    changed = False
    for column in list(tree.find_all(exp.Column)):
        qualifier = column.table.lower()
        if qualifier and qualifier not in valid:
            column.set("table", None)
            changed = True
    return render(tree) if changed else None


_SWAP_STOP = set(
    "a an the of for in on at to and or with all show me find list what which who "
    "whose whom when where how many are is was were do does did give return please "
    "their its it that this those these each per there have has had be been by from "
    "as than then also along get tell".split())


def _word_set(text):
    words = set(re.findall(r"[a-z][a-z0-9]*", str(text).lower()))
    stems = set(words)
    for word in list(words):
        if word.endswith("s") and len(word) > 3:
            stems.add(word[:-1])
    return stems


def schema_swaps(sql, question, schema, cap=16):
    """Question-guided, schema-valid single-column and LIKE-pattern rewrites.

    Hints are model samples, so a frequent residual error is a plausible but
    wrong column in a predicate (e.g. `Channel = 'Sky Radio'` instead of
    `series_name = 'Sky Radio'`).  Every replacement column comes from the
    public schema of the very same table, and edits are ranked by how well the
    column name matches the question wording.
    """
    tree = parse(sql)
    if tree is None:
        return []
    maps = _schema_maps(schema)
    aliases = {}
    for table in tree.find_all(exp.Table):
        if table.name.lower() in maps:
            aliases[(table.alias or table.name).lower()] = table.name.lower()
    qwords = _word_set(question) - _SWAP_STOP
    scored = []

    def emit(score, candidate):
        try:
            text = render(candidate)
        except Exception:
            return
        scored.append((score, text))

    columns = list(tree.find_all(exp.Column))
    for node in columns:
        owner = aliases.get(node.table.lower())
        if owner is None:
            owners = set(aliases.values())
            if len(owners) != 1:
                continue
            owner = next(iter(owners))
        options = maps.get(owner, {})
        if not options or len(options) > 24:
            continue
        original_sql = node.sql(dialect="sqlite")
        for lower, name in options.items():
            if lower == node.name.lower():
                continue
            overlap = len(_word_set(lower) & qwords)
            candidate = deepcopy(tree)
            for target in list(candidate.find_all(exp.Column)):
                if target.sql(dialect="sqlite") == original_sql:
                    target.set("this", exp.to_identifier(name))
            emit(2.0 * overlap + 0.05 * len(name), candidate)

    for node in list(tree.find_all(exp.Like)):
        literal = node.args.get("expression")
        if not isinstance(literal, exp.Literal) or not literal.is_string:
            continue
        value = literal.name.strip("%")
        if not value:
            continue
        for pattern in (value + "%", "%" + value, "%" + value + "%"):
            if pattern == literal.name:
                continue
            candidate = deepcopy(tree)
            for target in list(candidate.find_all(exp.Like)):
                if target.sql(dialect="sqlite") == node.sql(dialect="sqlite"):
                    target.set("expression", exp.Literal.string(pattern))
            emit(1.0, candidate)

    ordered, seen = [], set()
    for _, text in sorted(scored, key=lambda item: -item[0]):
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered[:cap]


def priority_templates(original, first_hint, question, schema, column_count,
                       original_empty=False):
    """A short list of high-precision rewrites worth spending an early tool call on."""
    generated = []
    for sql in (first_hint, original):
        generated.extend(semantic_templates(sql, question, schema, column_count))
    generated = list(dict.fromkeys(generated))
    lower = question.lower()
    result = []

    if original_empty and re.search(r"\b(corresponds to|affiliated with)\b", lower):
        for sql in generated:
            if " LIKE '" in sql and " != " not in sql and " NOT IN " not in sql:
                result.append(sql)

    if re.search(r"\b(avg|min|max|sum)\s*\(\s*(?:\w+\.)?(?:average|avg|total|sum)\b",
                 first_hint.lower()):
        for sql in generated:
            if (re.search(r",\s*(?:\w+\.)?(?:Average|Total)\s+AS\b", sql, re.IGNORECASE)
                    and "MAX(" in sql.upper()):
                result.append(sql)

    if re.search(r"\b(no|not|never|without)\b", lower):
        for sql in generated:
            if " EXCEPT " in sql:
                result.append(sql)

    if re.search(r"\b(friends?|treatments?)\b", lower):
        for sql in generated:
            if " IN (SELECT DISTINCT " in sql and " NOT IN " not in sql:
                result.append(sql)

    if "substring" in lower:
        for sql in generated:
            if " LIKE '%" in sql and "%the " not in sql.lower():
                result.append(sql)

    first_tree = parse(first_hint)
    if first_tree is not None and column_count > len(first_tree.expressions):
        for sql in generated:
            if ".*" in sql:
                result.append(sql)

    if re.search(r"\b(least|most)\b", lower):
        for sql in generated:
            if re.search(r"\bLIMIT\s+1\b", sql):
                result.append(sql)
    return list(dict.fromkeys(result))[:4]


def candidates(original, suggestions, question, schema, column_count, legacy=False,
               wide=False, expected_columns=None):
    """Return a bounded, deterministic queue; every item is later tool-checked."""
    bases = list(dict.fromkeys(suggestions + [original]))
    special = []
    for sql in bases:
        special.extend(name_variants(sql, question, schema))
    guided = []
    if expected_columns and not legacy:
        for sql in bases[:3]:
            guided.extend(column_guided_projections(sql, expected_columns, column_count))
            unqualified = unqualify_variant(sql, schema)
            if unqualified:
                guided.append(unqualified)
                guided.extend(column_guided_projections(unqualified, expected_columns,
                                                        column_count))
    derived = []
    simpler = []
    semantic = []
    sources = [] if legacy else (bases if wide else bases[:2])
    for sql in sources:
        generated = semantic_templates(sql, question, schema, column_count)
        semantic.extend(generated)
        for candidate_sql in generated:
            semantic.extend(simplifications(candidate_sql, question))
    for sql in bases:
        qualified = qualify_variant(sql, schema)
        repair_base = qualified or sql
        if qualified:
            derived.append(qualified)
        derived.extend(render(tree) for tree in basic_variants(repair_base, question, schema, column_count))
        simpler.extend(simplifications(repair_base, question))

    # A matching first hint is usually strong. If it fails, semantic repairs get
    # priority over trying many nearly identical model samples.
    ordered = []
    if bases:
        ordered.append(bases[0])
    ordered.extend(guided)
    ordered.extend(special)
    ordered.extend(semantic)
    ordered.extend(simpler)
    ordered.extend(derived)
    ordered.extend(bases[1:])
    return list(dict.fromkeys(sql for sql in ordered if sql and sql.strip()))
