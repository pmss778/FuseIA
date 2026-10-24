"""Final structured-policy datasets and model input preparation."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from policy_schema import NODE_KINDS, REL2ID


class StructuredDataset(Dataset):
    def __init__(self, directory, split, role):
        self.directory = Path(directory)
        self.role = role
        self.contexts = json.loads((self.directory / 'contexts.json').read_text(encoding='utf-8'))
        self.records = []
        for line in (self.directory / f'{split}.jsonl').open(encoding='utf-8'):
            if line.strip():
                record = json.loads(line)
                if record['role'] == role:
                    self.records.append(record)
        if not self.records:
            raise ValueError(f'No {role} records in {split}')
        keys = [r['sample_id'] for r in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate sample IDs')
        self.tokenized = {}

    def prepare(self, tokenizer, chunk_tokens=1024):
        for key, context in self.contexts.items():
            self.tokenized[key] = {}
            for view in ('original', 'anonymous'):
                schema = context['schema_' + view]
                chunks = []
                for qi, query in enumerate(context['queries_' + view]):
                    text = 'SCHEMA\n' + schema + '\nSQL\n' + query['sql']
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if not ids:
                        ids = [tokenizer.eos_token_id]
                    for offset in range(0, len(ids), chunk_tokens):
                        chunks.append({'ids': ids[offset:offset+chunk_tokens],
                                       'query_index': qi, 'offset': offset})
                self.tokenized[key][view] = chunks
        return self

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return record, self.contexts[record['context_id']], self.tokenized.get(record['context_id'])


def _index_ids(tokenizer, names):
    result = []
    for name in names:
        table, _, columns = name.partition('(')
        parts = []
        for column in columns.rstrip(')').split(','):
            text = f'{table}.{column.strip()}'
            ids = tokenizer.encode(text, add_special_tokens=False)
            parts.append(ids or [tokenizer.eos_token_id])
        result.append(parts)
    return result


def graph_input(graph, rng, hide_relations):
    n = len(graph['nodes'])
    if not n:
        raise ValueError('Empty SQL graph')
    positive = list(dict.fromkeys((int(s), int(t), relation) for s,t,relation in graph['edges']))
    # Sample targets before encoding; remove both directions and all other edge
    # labels for each held-out pair, so the answer is absent from attention.
    chosen = rng.sample(positive, min(16, max(1, len(positive)//6))) if positive else []
    selected_pairs = {(s,t) for s,t,_ in chosen}
    excluded = selected_pairs | {(t,s) for s,t in selected_pairs}
    existing = {(s,t) for s,t,_ in positive}
    targets = [(s,t,REL2ID[relation]) for s,t,relation in chosen]
    attempts = 0
    while len(targets) < 2*len(chosen) and attempts < max(100,n*4):
        attempts += 1
        s,t = rng.randrange(n),rng.randrange(n)
        if s != t and (s,t) not in existing and (t,s) not in existing:
            if not any(a==s and b==t for a,b,_ in targets):
                targets.append((s,t,REL2ID['none']))
    edges, types = [], []
    for s,t,relation in positive:
        if hide_relations and (s,t) in excluded:
            continue
        edges.extend([(s,t),(t,s)])
        types.extend([REL2ID[relation],REL2ID['rev:'+relation]])
    edges.extend((i,i) for i in range(n));types.extend([REL2ID['self']]*n)
    return {
        'kind':torch.tensor([NODE_KINDS.index(x['kind']) for x in graph['nodes']],dtype=torch.long),
        'numeric':torch.tensor([x['numeric'] for x in graph['nodes']],dtype=torch.float32),
        'text':[x['text'] for x in graph['nodes']],
        'edges':torch.tensor(edges,dtype=torch.long).T.contiguous(),
        'edge_types':torch.tensor(types,dtype=torch.long),
        'relation_pairs':torch.tensor([(s,t) for s,t,_ in targets],dtype=torch.long).reshape(-1,2),
        'relation_targets':torch.tensor([r for _,_,r in targets],dtype=torch.long),
    }


def prepare_pair(item, tokenizer, seed, training=False, chunk_tokens=1024):
    record, context, cached = item
    rng = random.Random(seed)
    graphs = [graph_input(g,rng,hide_relations=True) for g in context['graphs']]
    state = record['state']
    k = len(state['candidate_physical'])
    order = list(range(k))
    if training:
        rng.shuffle(order)
    inverse = {old:new for new,old in enumerate(order)}
    if record['role']=='proposer':
        labels = [inverse[x] for x in record['label_indices']]
        # A three-item shortlist ends by capacity, not a fourth sampled END.
        actions = labels + ([k] if len(labels)<3 else [])
    else:
        target = record['label_index']
        actions = [k if target==k else inverse[target]]
    output = {}
    for view in ('original','anonymous'):
        queries = context['queries_'+view]
        if cached:
            chunks=cached[view]
        else:
            chunks=[]
            for qi,q in enumerate(queries):
                ids=tokenizer.encode('SCHEMA\n'+context['schema_'+view]+'\nSQL\n'+q['sql'],add_special_tokens=False)
                ids=ids or [tokenizer.eos_token_id]
                for start in range(0,len(ids),chunk_tokens):
                    chunks.append({'ids':ids[start:start+chunk_tokens],'query_index':qi,'offset':start})
        names=[state['candidate_'+view][i] for i in order]
        output[view]={
            'semantic_chunks':chunks,'graphs':graphs,
            'candidate_ids':_index_ids(tokenizer,names),
            'current_ids':_index_ids(tokenizer,state['current_'+view]),
            'candidate_features':torch.tensor([state['candidate_features'][i] for i in order],dtype=torch.float32).reshape(k,-1)
                if k else torch.zeros((0,16),dtype=torch.float32),
            'candidate_node_links':[state['candidate_node_links'][i] for i in order],
            'state_features':torch.tensor(state['state_features'],dtype=torch.float32),
            'candidate_mask':torch.tensor([state['candidate_available'][i] for i in order],dtype=torch.bool),
            'feasible_mask':torch.tensor([state['candidate_feasible'][i] for i in order],dtype=torch.bool),
            'actions':actions,'candidate_physical':[state['candidate_physical'][i] for i in order],
            'sample_id':record['sample_id'],
        }
    return output


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data,dict):
        return {k:move_to_device(v,device) for k,v in data.items()}
    if isinstance(data,list):
        return [move_to_device(v,device) for v in data]
    return data
