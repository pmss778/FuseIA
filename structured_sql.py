"""SQL AST records with scope-local bindings and reversible input views."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]


@dataclass
class Predicate:
    table: str = ""
    column: str = ""
    operator: str = ""
    value_kind: str = "unknown"
    raw: str = ""
    block: str = "qb0"
    predicate_id: str = ""
    ref_ids: list[str] = field(default_factory=list)
    literal: Any = None
    literal_ids: list[str] = field(default_factory=list)
    shape: str = ""
    group: str = ""


@dataclass
class SqlStructure:
    parameterized_sql: str
    tables: list[str]
    aliases: dict[str, str]
    columns: list[str]
    predicates: list[Predicate]
    joins: list[dict]
    group_by: list[str]
    order_by: list[str]
    aggregates: list[str]
    query_blocks: int = 1
    blocks: list[dict] = field(default_factory=list)
    ctes: list[str] = field(default_factory=list)
    parser: str = "sqlglot"
    instances: list[dict] = field(default_factory=list)
    column_refs: list[dict] = field(default_factory=list)
    operations: list[dict] = field(default_factory=list)
    parameters: list[dict] = field(default_factory=list)
    raw_sql: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parameterize(tree):
    parameters = []
    for i, node in enumerate(list(tree.find_all(exp.Literal))):
        key = f"p{i}"
        value = node.this
        parameters.append({"id": key, "value": value,
                           "kind": "string" if node.is_string else "numeric"})
        parent = node.parent
        in_condition = False
        while parent is not None and not isinstance(parent, (exp.Select, exp.Union)):
            if isinstance(parent, (exp.Where, exp.Having, exp.Join)):
                in_condition = True
                break
            parent = parent.parent
        if not in_condition or isinstance(node.parent, (exp.Limit, exp.Offset, exp.Ordered, exp.Group)):
            continue
        node.replace(exp.Placeholder(this=key))
    return tree.sql(dialect="postgres"), parameters


def parameterize_sql(sql: str) -> str:
    return _parameterize(sqlglot.parse_one(sql, read="postgres"))[0]


def _schema_columns(schema) -> dict[str, set[str]]:
    if not schema:
        return {}
    if isinstance(schema, list):
        return {t["table"].lower(): {c["name"].lower() for c in t["columns"]} for t in schema}
    return {t.lower(): set(cs) if not isinstance(cs, dict) else set(cs) for t, cs in schema.items()}


def _inspect(sql: str, schema=None):
    tree = sqlglot.parse_one(sql, read="postgres")
    scopes = list(traverse_scope(tree))
    schema_cols = _schema_columns(schema)
    scope_ids = {id(s): f"qb{i}" for i, s in enumerate(scopes)}
    by_expr = {id(s.expression): s for s in scopes}
    instances, local = [], {}
    for s in scopes:
        bid = scope_ids[id(s)]
        local[id(s)] = {}
        for j, (alias, (_, source)) in enumerate(s.selected_sources.items()):
            rec = {"id": f"{bid}:i{j}", "block": bid, "alias": alias.lower(),
                   "table": source.name.lower() if isinstance(source, exp.Table) else "",
                   "source_block": scope_ids.get(id(source), ""),
                   "kind": "table" if isinstance(source, exp.Table) else "derived"}
            instances.append(rec)
            local[id(s)][alias.lower()] = (rec, source)

    def owner(node):
        cur = node
        while cur is not None:
            if id(cur) in by_expr:
                return by_expr[id(cur)]
            cur = cur.parent
        return scopes[-1] if scopes else None

    def resolve(scope, alias, name, visited=None):
        visited = set() if visited is None else visited
        if scope is None or (id(scope), alias, name) in visited:
            return None, []
        visited.add((id(scope), alias, name))
        sources = local[id(scope)]
        if alias:
            choices = [sources[alias]] if alias in sources else []
        else:
            choices = []
            for rec, source in sources.values():
                if isinstance(source, exp.Table):
                    if name in schema_cols.get(rec['table'], set()) or len(sources) == 1:
                        choices.append((rec, source))
                elif name in source.expression.named_selects:
                    choices.append((rec, source))
        if len(choices) != 1:
            return resolve(scope.parent, alias, name, visited) if not choices else (None, [])
        rec, source = choices[0]
        if isinstance(source, exp.Table):
            return rec, [f"{rec['table']}.{name}"]
        projections = list(source.expression.selects)
        names = source.outer_columns or source.expression.named_selects
        lineage = []
        if name in [x.lower() for x in names]:
            projection = projections[[x.lower() for x in names].index(name)]
            for c in projection.find_all(exp.Column):
                _, refs = resolve(source, c.table.lower(), c.name.lower(), set(visited))
                lineage.extend(refs)
        return rec, sorted(set(lineage))

    columns = list(tree.find_all(exp.Column))
    refs, node_refs = [], {}
    for i, c in enumerate(columns):
        s = owner(c); bid = scope_ids.get(id(s), "qb0")
        rec, physical = resolve(s, c.table.lower(), c.name.lower())
        ref = {"id": f"r{i}", "block": bid, "instance": rec['id'] if rec else "",
               "name": c.name.lower(), "qualifier": c.table.lower(),
               "physical_columns": physical, "bound": bool(rec),
               "symbol": (physical[0] if rec and rec['kind']=='table' else
                          f"{rec['source_block'] if rec else bid}.{c.name.lower()}")}
        refs.append(ref); node_refs[id(c)] = ref
    return tree, scopes, scope_ids, instances, refs, node_refs, owner


def parse_sql_structure(sql: str, schema=None) -> SqlStructure:
    tree, scopes, scope_ids, instances, refs, node_refs, owner = _inspect(sql, schema)
    blocks = []
    for s in scopes:
        depth, parent = 0, s.parent
        while parent is not None:
            depth += 1; parent = parent.parent
        blocks.append({"id": scope_ids[id(s)], "parent": scope_ids.get(id(s.parent)),
                       "kind": s.scope_type.name.lower(), "depth": depth,
                       "name": (s.expression.parent.alias or '').lower()
                       if isinstance(s.expression.parent, exp.CTE) else ""})
    literal_nodes = list(tree.find_all(exp.Literal))
    literal_ids = {id(n): f"p{i}" for i, n in enumerate(literal_nodes)}
    predicates, joins, operations = [], [], []
    op_names = {exp.EQ: '=', exp.NEQ: '<>', exp.LT: '<', exp.LTE: '<=', exp.GT: '>', exp.GTE: '>=',
                exp.Like: 'LIKE', exp.ILike: 'ILIKE', exp.In: 'IN', exp.Between: 'BETWEEN', exp.Is: 'IS'}
    def ref_list(node):
        return [node_refs[id(c)] for c in node.find_all(exp.Column) if id(c) in node_refs]
    def first_physical(rs):
        return next((x for r in rs for x in r['physical_columns']), '')
    for node in tree.walk():
        if type(node) not in op_names:
            continue
        op = op_names[type(node)]
        rs = ref_list(node); s = owner(node); bid = scope_ids.get(id(s), 'qb0')
        if not rs:
            continue
        if isinstance(node, (exp.EQ,exp.NEQ,exp.LT,exp.LTE,exp.GT,exp.GTE)) and isinstance(node.left,exp.Column) and isinstance(node.right,exp.Column):
            l,r = node_refs[id(node.left)],node_refs[id(node.right)]
            joins.append({'left':l['symbol'],'right':r['symbol'],'left_ref':l['id'],'right_ref':r['id'],
                          'operator':op,'block':bid})
            continue
        lit = [n for n in node.find_all(exp.Literal)]
        value = [n.this for n in lit] if isinstance(node,(exp.In,exp.Between)) else (lit[0].this if lit else None)
        kind = 'list' if isinstance(node,exp.In) else ('string' if lit and lit[0].is_string else 'numeric' if lit else 'unknown')
        shape = ''
        if isinstance(node,(exp.Like,exp.ILike)) and value is not None:
            shape = 'leading_wildcard' if str(value).startswith(('%','_')) else 'prefix' if str(value).endswith('%') else 'exact'
        if isinstance(node,exp.Is) and isinstance(node.expression,exp.Null):
            op = 'IS NOT NULL' if isinstance(node.parent,exp.Not) else 'IS NULL'; kind='null'
        elif isinstance(node.parent,exp.Not):
            op='NOT '+op
        if isinstance(node,(exp.LT,exp.LTE,exp.GT,exp.GTE)) and not isinstance(node.left,exp.Column):
            op={'<':'>','<=':'>=','>':'<','>=':'<='}[op]
        physical=first_physical(rs);tab,_,col=physical.partition('.')
        predicates.append(Predicate(tab,col,op,kind,node.sql(dialect='postgres'),bid,
                                    f'pr{len(predicates)}',[r['id'] for r in rs],value,
                                    [literal_ids[id(n)] for n in lit],shape))
    group_by,order_by,aggregates=[],[],[]
    for node in tree.walk():
        kind = 'group_by' if isinstance(node,exp.Group) else 'order_by' if isinstance(node,exp.Order) else 'aggregate' if isinstance(node,exp.AggFunc) else ''
        if not kind: continue
        rs=ref_list(node); bid=scope_ids.get(id(owner(node)),'qb0')
        operators = []
        if kind == 'order_by': operators=['DESC' if o.args.get('desc') else 'ASC' for o in node.expressions]
        func=type(node).__name__.lower() if kind=='aggregate' else kind
        operations.append({'id':f'op{len(operations)}','kind':kind,'block':bid,
                           'refs':[r['id'] for r in rs],'operator':func,'directions':operators})
        values=[r['symbol'] for r in rs]
        if kind=='group_by':group_by.extend(values)
        elif kind=='order_by':order_by.extend(values)
        else:aggregates.append(f"{func}({','.join(values) if values else '*'})")
    parameterized,params=_parameterize(tree.copy())
    tables=sorted({r['table'] for r in instances if r['table']})
    aliases={r['alias']:r['table'] or r['source_block'] for r in instances}
    return SqlStructure(parameterized,tables,aliases,sorted({r['symbol'] for r in refs}),predicates,joins,
                        group_by,order_by,aggregates,len(scopes),blocks,
                        sorted({b['name'] for b in blocks if b['name']}),'sqlglot',
                        instances,refs,operations,params,sql)


def rename_ast_sql(sql: str, mapping: dict[str,str], schema=None) -> str:
    tree,scopes,scope_ids,instances,refs,node_refs,owner=_inspect(sql,schema)
    instance_alias = {r['id']:f"a{r['id'].replace('qb','').replace(':i','_')}" for r in instances}
    # Rewrite references while binding still sees the original AST.
    for c in list(tree.find_all(exp.Column)):
        ref=node_refs[id(c)]
        mapped=mapping.get(ref['symbol'],ref['symbol'])
        c.set('this',exp.to_identifier(mapped.rsplit('.',1)[-1]))
        if ref['instance']:
            c.set('table',exp.to_identifier(instance_alias[ref['instance']]))
    for s in scopes:
        bid=scope_ids[id(s)]
        for projection in s.expression.selects:
            if isinstance(projection,exp.Alias):
                symbol=f'{bid}.{projection.alias.lower()}'
                projection.set('alias',exp.to_identifier(mapping.get(symbol, f'{bid}.o{list(s.expression.selects).index(projection)}').rsplit('.',1)[-1]))
        if isinstance(s.expression.parent,exp.CTE):
            cte=s.expression.parent
            cte.set('alias',exp.TableAlias(this=exp.to_identifier(mapping.get(cte.alias.lower(),cte.alias.lower()))))
        selected=list(s.selected_sources.items())
        for j, (alias,(node,source)) in enumerate(selected):
            iid=f'{bid}:i{j}'
            alias_node=exp.TableAlias(this=exp.to_identifier(instance_alias[iid]))
            if isinstance(source,exp.Table):
                node.set('this',exp.to_identifier(mapping.get(source.name.lower(),source.name.lower())))
                node.set('alias',alias_node)
            else:
                if isinstance(node,exp.Table): node.set('this',exp.to_identifier(mapping.get(node.name.lower(),node.name.lower())))
                if isinstance(node,exp.Subquery):node.set('alias',alias_node)
                else:node.set('alias',alias_node)
    return _parameterize(tree)[0]
