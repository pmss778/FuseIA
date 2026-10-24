"""Shared vocabularies and feature schema for the dual-path policy."""
NODE_KINDS = ['query_block','table','table_instance','column','column_reference',
              'predicate','join','aggregate','group_by','order_by']
EDGE_RELATIONS = ['contains_table','contains_table_instance','instance_of',
                  'instance_of_block','owns_column','applies_to','contains_predicate',
                  'contains_join','left','right','uses','contains_aggregate',
                  'contains_block','references','bound_to','contains_reference',
                  'contains_operation']
RELATIONS = ['none','self'] + EDGE_RELATIONS + ['rev:' + r for r in EDGE_RELATIONS]
REL2ID = {r: i for i,r in enumerate(RELATIONS)}
NODE_NUM_FEATURES = ['match_fraction','match_valid','log_rows','log_ndv','null_fraction',
                     'stats_valid','log_match_rows','depth','value_count','ascending']
# Schema v3 (2026-10-14): 12 dimensions, all syntax facts (parsed from the
# workload SQL) or storage-catalog facts (pg_class/pg_stats metadata), plus
# episode protocol state. No plan-time estimates (rows_kept_fraction /
# plan_rows / planner ndv guesses were identified as false-signal sources --
# see EXPERIMENT_TRACKER 2026-10-14). Join identity comes from
# candidate_pool.OfflineStore._join_profile (cross-table comparisons).
CAND_FEATURES = ['log_bytes',            # 1 index storage cost
                 'log_rows',             # 2 table scale (catalog)
                 'null_frac',            # 3 column null rate (catalog snapshot)
                 'selected',             # 4 protocol: already in set
                 'feasible',             # 5 protocol: fits remaining budget
                 'distinct_query_count', # 6 column-level total heat (queries)
                 'filter_count',         # 7 predicate references (strength)
                 'join_count',           # 8 cross-table comparisons (queries)
                 'eq_range_ratio',       # 9 equality/range operator profile
                 'filter_query_count',   # 10 distinct queries filtering on it
                 'join_query_ratio',     # 11 join share of all references
                 'table_query_count']    # 12 table-level heat (queries)
STATE_FEATURES = ['log_budget','used_fraction','remaining_fraction','log_selected',
                  'log_pool','log_feasible']
FEATURE_SCHEMA = {'node_kinds': NODE_KINDS,'relations': RELATIONS,
                  'node_numeric':NODE_NUM_FEATURES,'candidate':CAND_FEATURES,
                  'state':STATE_FEATURES}
