"""Shared SQL, graph, statistics and state inputs for teacher/student policies."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from candidate_pool import OfflineStore
from predicate_stats import estimate_predicate
from relational_graph import make_rename_map, anonymize_index
from structured_sql import parse_sql_structure, rename_ast_sql


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schema_index(schema):
    return {f"{table['table']}.{col['name']}": col['type']
            for table in schema for col in table['columns']}


def _ndv(stats, physical):
    table, _, column = physical.partition('.')
    item = stats['columns'].get((table, column), {})
    value = item.get('n_distinct')
    rows = stats['rows'].get(table)
    if value is not None and value < 0:
        value = -value * rows if rows else None
    return value, rows, item


def _node_numbers(evidence=None, depth=0, count=0, ascending=0):
    ev = evidence or {}
    fraction = ev.get('estimated_match_fraction')
    rows, ndv = ev.get('table_rows'), ev.get('n_distinct')
    return [float(fraction or 0), float(fraction is not None),
            math.log1p(max(float(rows or 0), 0)), math.log1p(max(float(ndv or 0), 0)),
            float(ev.get('null_frac') or 0), float(rows is not None or ndv is not None),
            math.log1p(max(float(ev.get('estimated_matched_rows') or 0), 0)),
            float(depth), math.log1p(count), float(ascending)]


def build_context(queries, pool_rows, schema, stats, seed, rename_map=None):
    """One full workload. Physical names are metadata, never graph text features."""
    types = schema_index(schema)
    structures = [parse_sql_structure(q['sql'], schema) for q in queries]
    extras = []
    for row in pool_rows:
        table, _, cs = row['name'].partition('(')
        extras.append(table)
        extras.extend(f'{table}.{c.strip()}' for c in cs.rstrip(')').split(','))
    # Include schema so columns displayed with candidate descriptions also map.
    extras.extend(types)
    mapping = rename_map or make_rename_map(structures, seed, extras)
    original, anonymous, graph_queries = [], [], []
    roles = defaultdict(Counter)
    fractions = defaultdict(list)
    evidence_rows = []
    for qi, (query, structure) in enumerate(zip(queries, structures)):
        original.append({'query_id': query['query_id'], 'sql': structure.parameterized_sql})
        anonymous.append({'query_id': query['query_id'],
                          'sql': rename_ast_sql(query['sql'], mapping, schema)})
        nodes, edges, column_nodes = [], [], defaultdict(list)
        id_map = {}
        def add(nid, kind, text, numeric=None):
            id_map[nid] = len(nodes)
            nodes.append({'id': nid, 'kind': kind, 'text': text,
                          'numeric': numeric if numeric is not None else _node_numbers()})
        def edge(source, target, relation):
            if source in id_map and target in id_map:
                edges.append([id_map[source], id_map[target], relation])
        for block in structure.blocks:
            add(block['id'], 'query_block', 'query '+block['kind'],
                _node_numbers(depth=block['depth']))
        for block in structure.blocks:
            if block['parent']: edge(block['parent'], block['id'], 'contains_block')
        for instance in structure.instances:
            add(instance['id'], 'table_instance', 'table_instance')
            edge(instance['block'], instance['id'], 'contains_table_instance')
            if instance['source_block']:edge(instance['id'],instance['source_block'],'instance_of_block')
        for ri, ref in enumerate(structure.column_refs):
            physical = ref['physical_columns']
            dtype = types.get(physical[0], 'unknown') if physical else 'unknown'
            nid = ref['id']
            add(nid, 'column_reference', 'column '+dtype)
            edge(ref['block'], nid, 'contains_reference')
            if ref['instance']:edge(nid, ref['instance'], 'bound_to')
            for column in physical: column_nodes[column].append(id_map[nid])
        for predicate in structure.predicates:
            ev = estimate_predicate(predicate, stats)
            safe = {k:v for k,v in ev.items() if k not in ('literal',)}
            evidence_rows.append({'query_id':query['query_id'],**safe})
            nid = predicate.predicate_id
            value_count = len(predicate.literal) if isinstance(predicate.literal,list) else 0
            text = f"predicate {predicate.operator} {predicate.value_kind} {predicate.shape} {ev['source']}"
            add(nid, 'predicate', text, _node_numbers(ev, count=value_count))
            edge(predicate.block,nid,'contains_predicate')
            for refid in predicate.ref_ids: edge(nid,refid,'applies_to')
            physical = f'{predicate.table}.{predicate.column}'
            roles[physical]['filter'] += 1
            if ev['estimated_match_fraction'] is not None:
                fractions[physical].append(ev['estimated_match_fraction'])
        refs = {ref['id']:ref for ref in structure.column_refs}
        for ji, join in enumerate(structure.joins):
            nid = f'join:{ji}'
            add(nid,'join','join '+join['operator'])
            edge(join['block'],nid,'contains_join')
            edge(nid,join['left_ref'],'left');edge(nid,join['right_ref'],'right')
            for refid in (join['left_ref'],join['right_ref']):
                for physical in refs[refid]['physical_columns']:roles[physical]['join']+=1
        for operation in structure.operations:
            nid=operation['id'];kind=operation['kind']
            directions=' '.join(operation.get('directions',[]))
            add(nid,kind,kind+' '+operation['operator']+' '+directions,
                _node_numbers(ascending=float('ASC' in directions)))
            edge(operation['block'],nid,'contains_operation')
            for refid in operation['refs']:
                edge(nid,refid,'uses')
                for physical in refs[refid]['physical_columns']:
                    roles[physical][{'group_by':'group','order_by':'order'}.get(kind,kind)]+=1
        graph_queries.append({'nodes':nodes,'edges':edges,
                              'column_nodes':dict(column_nodes),'query_id':query['query_id']})
    symbols=set()
    for structure in structures: symbols.update(structure.columns)
    for row in pool_rows:
        table, _, cols=row['name'].partition('(')
        symbols.update(f'{table}.{col.strip()}' for col in cols.rstrip(')').split(','))
    schema_original=[];schema_anon=[]
    for physical in sorted(symbols):
        dtype=types.get(physical,'unknown')
        schema_original.append(f'{physical}: {dtype}')
        schema_anon.append(f'{mapping.get(physical,physical)}: {dtype}')
    return {'queries_original':original,'queries_anonymous':anonymous,
            'schema_original':'\n'.join(schema_original),
            'schema_anonymous':'\n'.join(schema_anon),'graphs':graph_queries,
            'roles':{p:dict(v) for p,v in roles.items()},
            'fractions':dict(fractions),'rename_map':mapping,'stats_version':stats['stats_version'],
            'predicate_evidence':evidence_rows,
            'execution_queries':queries}




_CONTEXT_PROFILE_CACHE = {}


def context_reference_profile(context, pool_rows):
    """Aggregate per-column syntax facts from the parsed workload (cached).

    All values are syntax facts from parsed SQL -- no plan-time estimates.
    join counts reuse candidate_pool.OfflineStore._join_profile.
    Returns ({'table.column': {...}}, total_query_count).
    """
    cache_key = id(context)
    if cache_key in _CONTEXT_PROFILE_CACHE:
        return _CONTEXT_PROFILE_CACHE[cache_key]
    queries = context.get('execution_queries') or []
    total = len(queries)
    try:
        import sqlglot
        from sqlglot import exp
        from sqlglot.optimizer.scope import traverse_scope
    except ImportError:
        result = ({}, total)
        _CONTEXT_PROFILE_CACHE[cache_key] = result
        return result
    # column-name -> owning tables from the frozen pool (resolves
    # unqualified column references like TPC-H's bare l_shipdate)
    column_owners = {}
    for row in pool_rows:
        column_owners.setdefault(str(row['column']).lower(), set()).add(
            str(row['table']).lower())
    col_queries = {}
    table_queries = {}
    filter_queries = {}
    op_eq = {}
    op_range = {}
    parsed = 0
    for query in queries:
        try:
            tree = sqlglot.parse_one(query['sql'], read='postgres')
        except Exception:
            continue
        parsed += 1
        alias_map = {}
        for scope in traverse_scope(tree):
            for name, source in scope.sources.items():
                if isinstance(source, exp.Table):
                    alias_map.setdefault(name.lower(),
                                         set()).add(source.name.lower())
        seen_cols = set()
        seen_tables = set()
        filter_cols = set()
        eq_count = {}
        range_count = {}

        def resolve(node):
            ref = (node.table or '').lower()
            tables = alias_map.get(ref, {ref} if ref else set())
            exact = sorted(t for t in tables
                           if not t.startswith('__subquery__'))
            if not exact and not ref:
                owners = column_owners.get(node.name.lower(), set())
                exact = sorted(owners) if len(owners) == 1 else []
            return exact

        for node in tree.find_all(exp.Column):
            exact = resolve(node)
            if not exact:
                continue
            seen_cols.add(f'{exact[0]}.{node.name.lower()}')
            seen_tables.add(exact[0])
        for predicate in tree.find_all(exp.Predicate):
            cols = list(predicate.find_all(exp.Column))
            if not cols:
                continue
            family = {exp.EQ: 'eq', exp.In: 'eq', exp.Is: 'eq',
                      exp.GT: 'range', exp.GTE: 'range',
                      exp.LT: 'range', exp.LTE: 'range',
                      exp.Between: 'range'}.get(type(predicate).__mro__[0]
                                                if False else type(predicate))
            if type(predicate).__name__ in ('EQ', 'In', 'Is'):
                family = 'eq'
            elif type(predicate).__name__ in ('GT', 'GTE', 'LT', 'LTE',
                                              'Between'):
                family = 'range'
            elif type(predicate).__name__ in ('Like', 'ILike'):
                family = 'like'
            else:
                family = None
            if family is None:
                continue
            for c in cols:
                exact = resolve(c)
                if not exact:
                    continue
                key = f'{exact[0]}.{c.name.lower()}'
                if family == 'eq':
                    eq_count[key] = eq_count.get(key, 0) + 1
                elif family == 'range':
                    range_count[key] = range_count.get(key, 0) + 1
                filter_cols.add(key)
        for key in seen_cols:
            col_queries.setdefault(key, set()).add(id(query))
        for key in seen_tables:
            table_queries.setdefault(key, set()).add(id(query))
        for key in filter_cols:
            filter_queries.setdefault(key, set()).add(id(query))
    from candidate_pool import OfflineStore
    probe = OfflineStore('profile', pool_rows, 1.0)
    join_counts, _ = probe._join_profile(queries)
    profile = {}
    for key in (set(col_queries) | set(join_counts) | set(op_eq)
                | set(op_range) | set(op_eq) | set(op_range)):
        dc = len(col_queries.get(key, set()))
        j = join_counts.get(key, 0)
        eq = eq_count.get(key, 0)
        rng = range_count.get(key, 0)
        profile[key] = {
            'distinct_query_count': dc,
            'filter_query_count': len(filter_queries.get(key, set())),
            'join_count': j,
            'eq_range_ratio': (eq / rng) if rng else (1.0 if eq else 0.5),
        }
    result = (profile, parsed or total)
    _CONTEXT_PROFILE_CACHE[cache_key] = result
    return result


def _column_filter_count(context, physical):
    """Predicate reference count (v1 'filter' semantics): occurrences of the
    column in predicates, from context roles (syntax facts)."""
    counts = Counter()
    for p in physical:
        counts.update(context['roles'].get(p, {}))
    return counts.get('filter', 0)


def _table_query_count(context, table):
    """Distinct queries referencing the table (any column)."""
    from sqlglot import exp
    seen = 0
    queries = context.get('execution_queries') or []
    try:
        import sqlglot
        from sqlglot.optimizer.scope import traverse_scope
    except ImportError:
        return 0
    for query in queries:
        try:
            tree = sqlglot.parse_one(query['sql'], read='postgres')
        except Exception:
            continue
        tables = set()
        for scope in traverse_scope(tree):
            for name, source in scope.sources.items():
                if isinstance(source, exp.Table):
                    tables.add(source.name.lower())
        if table.lower() in tables:
            seen += 1
    return seen


def _table_rows(stats, physical, rows, name):
    """Table row count from predicate-stats snapshot; fall back to the
    frozen pool's catalog value (table_rows is catalog data, not a plan
    estimate, so the fallback is safe unlike n_distinct)."""
    _, nrows, colstats = _ndv(stats, physical[0])
    pg = rows[name].get('pg_stats') or {}
    if nrows is None:
        nrows = pg.get('table_rows')
    return nrows, colstats


def _table_of(name, rows):
    row = rows.get(name) or {}
    return str(row.get('table', name.partition('(')[0])).lower()


def build_state(context, pool_rows, stats, current, budget, candidates=None):
    """State features use physical IDs; both views receive identical numeric facts."""
    store=OfflineStore('offline',pool_rows,budget)
    names=candidates if candidates is not None else sorted(store.rows)
    used=store.estimate_storage_mib(current)
    remaining=max(0.,budget-used)
    rows=store.rows
    features=[];node_links=[]
    for name in names:
        table,_,cs=name.partition('(');columns=[c.strip() for c in cs.rstrip(')').split(',')]
        physical=[f'{table}.{col}' for col in columns]
        size=float(rows[name]['storage_mib'])
        nrows, colstats=_table_rows(stats,physical[0],rows,name)
        profile, total_q = context_reference_profile(context, pool_rows)
        key = physical[0]
        ref = profile.get(key, {})
        dqc = ref.get('distinct_query_count', 0)
        j = ref.get('join_count', 0)
        fqc = ref.get('filter_query_count', 0)
        eq_ratio = ref.get('eq_range_ratio', 1.0)
        features.append([math.log1p(size),
                         math.log1p(max(nrows or 0, 0)),
                         float(colstats.get('null_frac', (rows[name].get('pg_stats') or {}).get('null_frac', 0))),
                         float(name in current),
                         float(name not in current and size <= remaining + 1e-6),
                         float(dqc) / max(total_q, 1),
                         float(_column_filter_count(context, physical)),
                         float(j) / max(total_q, 1),
                         float(eq_ratio),
                         float(fqc) / max(total_q, 1),
                         min(float(j) / max(dqc, 1), 1.0) if dqc else 0.0,
                         float(_table_query_count(context, table)) / max(total_q, 1)])
        links=[]
        for qi, graph in enumerate(context['graphs']):
            for order,p in enumerate(physical):
                links.extend([qi,index,order] for index in graph['column_nodes'].get(p,[]))
        node_links.append(links)
    mapping=context['rename_map']
    result={'candidate_physical':names,'candidate_original':names,
            'candidate_anonymous':[anonymize_index(n,mapping) for n in names],
            'candidate_features':features,'candidate_node_links':node_links,
            'candidate_available':[True]*len(names),
            'candidate_feasible':[bool(x[4]) for x in features],
            'current_original':list(current),
            'current_anonymous':[anonymize_index(n,mapping) for n in current],
            'state_features':[math.log1p(budget),used/max(budget,1e-6),remaining/max(budget,1e-6),
                              math.log1p(len(current)),math.log1p(len(names)),
                              math.log1p(sum(bool(x[4]) for x in features))],
            'budget_mib':budget,'storage_used_mib':used,'storage_left_mib':remaining}
    return result


def teacher_queries(context):
    """Serialize the same non-execution information a dual-path student receives."""
    queries=[dict(q) for q in context['queries_original']]
    # The prompt contract is unchanged. Evidence is contextual input appended as SQL comments.
    facts=[]
    for row in context['predicate_evidence']:
        facts.append({'q':row['query_id'],'column':row['table']+'.'+row['column'],
                      'operator':row['operator'],'match_fraction':row['estimated_match_fraction'],
                      'source':row['source']})
    queries.append({'query_id':'structural_evidence',
                    'sql':'-- PREDICATE STATISTICS\n-- '+json.dumps(facts,ensure_ascii=False,
                                                                          separators=(',',':'))})
    return queries
