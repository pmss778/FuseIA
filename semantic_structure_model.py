"""Role-specific semantic/relational policy used by SFT, inference and GRPO."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from policy_schema import NODE_KINDS, RELATIONS, REL2ID, CAND_FEATURES, NODE_NUM_FEATURES, STATE_FEATURES


@dataclass
class ModelConfig:
    backbone_name: str
    role: str
    d_model: int = 256
    n_heads: int = 8
    graph_layers: int = 3
    fusion_layers: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    dropout: float = 0.1
    chunk_tokens: int = 1024
    max_shortlist: int = 3
    gradient_checkpointing: bool = True
    precision: str = 'bf16'


class SymbolEncoder(nn.Module):
    """Deterministic embeddings for name-free structural descriptors."""
    def __init__(self, dim, buckets=4096):
        super().__init__()
        self.embedding = nn.Embedding(buckets, dim)
        self.buckets = buckets

    def forward(self, texts):
        output = []
        for text in texts:
            words = str(text).replace('_', ' ').split() or ['empty']
            ids = [int.from_bytes(hashlib.sha256(w.encode()).digest()[:8], 'little') % self.buckets
                   for w in words]
            indices = torch.tensor(ids, device=self.embedding.weight.device)
            output.append(self.embedding(indices).mean(0))
        return torch.stack(output)


class SemanticBackbone(nn.Module):
    def __init__(self, config: ModelConfig, tiny_config=None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.config = config
        if tiny_config is not None:
            self.hf = AutoModel.from_config(tiny_config)
            self.tokenizer = None
        else:
            dtype = torch.bfloat16 if config.precision == 'bf16' else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
            self.hf = AutoModel.from_pretrained(config.backbone_name, torch_dtype=dtype,
                                               attn_implementation='sdpa')
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,
                          lora_dropout=config.lora_dropout, bias='none',
                          target_modules=['q_proj','k_proj','v_proj','o_proj',
                                          'gate_proj','up_proj','down_proj'])
        self.hf = get_peft_model(self.hf, lora)
        self.hf.config.use_cache = False
        if config.gradient_checkpointing:
            self.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self.hf.enable_input_require_grads()
        self.hidden_size = self.hf.config.hidden_size
        self.projection = nn.Linear(self.hidden_size, config.d_model)
        self.name_projection = nn.Linear(self.hidden_size, config.d_model)

    def _chunk(self, ids, offset, query_index):
        device = self.projection.weight.device
        input_ids = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
        positions = torch.arange(offset, offset + len(ids), device=device).unsqueeze(0)
        hidden = self.hf(input_ids=input_ids, position_ids=positions,
                         attention_mask=torch.ones_like(input_ids), use_cache=False).last_hidden_state[0]
        projected = self.projection(hidden.float())
        # Fixed query-position encoding separates repeated SQL blocks without names or labels.
        dimension = torch.arange(projected.shape[-1], device=device, dtype=torch.float32)
        scale = 10000.0 ** (-2 * torch.floor(dimension / 2) / projected.shape[-1])
        phase = (query_index + 1) * scale
        query_pos = torch.where(dimension.long() % 2 == 0, phase.sin(), phase.cos())
        return projected + query_pos

    def forward(self, chunks):
        tokens = []
        for item in chunks:
            ids, offset, qi = item['ids'], item['offset'], item['query_index']
            if self.training and self.config.gradient_checkpointing:
                # Checkpoint projection together with HF so all query chunks can be retained.
                def fn(anchor, ids=ids, offset=offset, qi=qi):
                    return self._chunk(ids, offset, qi) + anchor * 0
                anchor = self.projection.weight.new_zeros((), requires_grad=True)
                tokens.append(checkpoint(fn, anchor, use_reentrant=False))
            else:
                tokens.append(self._chunk(ids, offset, qi))
        if not tokens:
            raise ValueError('Semantic input has no tokens')
        return torch.cat(tokens, 0)

    def encode_names(self, sequences):
        # Candidate and current-index identities use pretrained token embeddings;
        # full schema/SQL contextual semantics arrive through the backbone above.
        embedding = self.hf.get_input_embeddings()
        device = embedding.weight.device
        vectors = []
        for columns in sequences:
            parts = []
            for order, ids in enumerate(columns):
                index = torch.tensor(ids or [0], device=device)
                value = self.name_projection(embedding(index).mean(0).float())
                dimension = torch.arange(value.shape[-1], device=device).float()
                value = value + torch.sin((order + 1) / (10000 ** (dimension / value.shape[-1])))
                parts.append(value)
            vectors.append(torch.stack(parts).mean(0))
        return torch.stack(vectors) if vectors else self.name_projection.weight.new_zeros((0,self.config.d_model))


class RelationAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)
        self.bias = nn.Embedding(len(RELATIONS), heads)
        nn.init.zeros_(self.bias.weight)
        self.dropout = dropout

    def forward(self, x, edges, types):
        n, dim = x.shape
        q,k,v = self.qkv(x).view(n,3,self.heads,self.head_dim).permute(1,2,0,3)
        bias = x.new_zeros((self.heads,n,n))
        if types.numel():
            values = self.bias(types).transpose(0,1)
            flat_index = edges[0] * n + edges[1]
            bias = bias.reshape(self.heads,-1).scatter_add(1,flat_index.unsqueeze(0).expand(self.heads,-1),values).view(self.heads,n,n)
        # Edges can carry multiple relation types; their learned biases are summed.
        out = F.scaled_dot_product_attention(q,k,v,attn_mask=bias,
                                             dropout_p=self.dropout if self.training else 0.)
        return self.output(out.transpose(0,1).reshape(n,dim))


class RelationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d=config.d_model
        self.norm1=nn.LayerNorm(d); self.norm2=nn.LayerNorm(d)
        self.attention=RelationAttention(d,config.n_heads,config.dropout)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(4*d,d))
        self.drop=nn.Dropout(config.dropout)

    def forward(self,x,edges,types):
        x=x+self.drop(self.attention(self.norm1(x),edges,types))
        return x+self.drop(self.ff(self.norm2(x)))


class RelationGraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d=config.d_model
        self.kinds=nn.Embedding(len(NODE_KINDS),d)
        self.descriptors=SymbolEncoder(d)
        self.numeric=nn.Linear(len(NODE_NUM_FEATURES),d)
        self.mix=nn.Linear(3*d,d)
        self.layers=nn.ModuleList(RelationLayer(config) for _ in range(config.graph_layers))
        self.norm=nn.LayerNorm(d)

    def forward(self, graph):
        x=self.mix(torch.cat([self.kinds(graph['kind']),self.descriptors(graph['text']),
                              self.numeric(graph['numeric'])],-1))
        for layer in self.layers:
            x=layer(x,graph['edges'],graph['edge_types'])
        return self.norm(x)


class SemanticStructureModel(nn.Module):
    """Each role has an independent LoRA, graph encoder, fusion and action head."""
    def __init__(self, config: ModelConfig, tiny_config=None):
        super().__init__()
        if config.role not in ('proposer','arbiter'):
            raise ValueError(config.role)
        self.config=config
        self.backbone=SemanticBackbone(config,tiny_config)
        self.graph=RelationGraphEncoder(config)
        d=config.d_model
        self.candidate_numeric=nn.Sequential(nn.Linear(len(CAND_FEATURES),d),nn.GELU(),nn.Linear(d,d))
        self.grounded=nn.Linear(d,d)
        self.state_encoder=nn.Sequential(nn.Linear(len(STATE_FEATURES),d),nn.GELU(),nn.Linear(d,d))
        self.context_types=nn.Embedding(5,d)
        self.terminal_token=nn.Parameter(torch.zeros(1,d))
        self.fusion=nn.ModuleList(nn.TransformerDecoderLayer(
            d,config.n_heads,4*d,dropout=config.dropout,batch_first=True,
            norm_first=True) for _ in range(config.fusion_layers))
        self.fusion_norm=nn.LayerNorm(d)
        self.initial=nn.Linear(d,d)
        self.pointer=nn.Linear(d,d,bias=False)
        self.prefix_update=nn.GRUCell(d,d)
        self.step_embedding=nn.Embedding(config.max_shortlist,d)
        self.stop_head=nn.Linear(d,1)
        self.rel_head=nn.Linear(2*d,len(RELATIONS))

    def encode(self,batch):
        semantic=self.backbone(batch['semantic_chunks'])
        nodes=[self.graph(g) for g in batch['graphs']]
        state=self.state_encoder(batch['state_features']).unsqueeze(0)
        current=self.backbone.encode_names(batch['current_ids'])
        numeric=self.candidate_numeric(batch['candidate_features'])
        candidates=self.backbone.encode_names(batch['candidate_ids']) + numeric
        if candidates.shape[0]:
            grounded=[]
            for links in batch['candidate_node_links']:
                terms=[nodes[qi][ni] for qi,ni,order in links]
                grounded.append(torch.stack(terms).mean(0) if terms else numeric.new_zeros(self.config.d_model))
            candidates=candidates+self.grounded(torch.stack(grounded))
        graph_memory=torch.cat(nodes,0) if nodes else state.new_zeros((0,self.config.d_model))
        memory=torch.cat([semantic+self.context_types.weight[0],
                           graph_memory+self.context_types.weight[1],
                           current+self.context_types.weight[2],state+self.context_types.weight[3]],0)
        queries=torch.cat([candidates,self.terminal_token+state],0).unsqueeze(0)
        for layer in self.fusion:
            queries=layer(queries,memory.unsqueeze(0))
        fused=self.fusion_norm(queries[0])
        pairs=[]; labels=[]
        for g,representation in zip(batch['graphs'],nodes):
            if g['relation_pairs'].numel():
                left,right=g['relation_pairs'][:,0],g['relation_pairs'][:,1]
                pairs.append(torch.cat([representation[left],representation[right]],-1))
                labels.append(g['relation_targets'])
        rel_logits=self.rel_head(torch.cat(pairs)) if pairs else fused.new_zeros((0,len(RELATIONS)))
        rel_targets=torch.cat(labels) if labels else torch.empty(0,device=fused.device,dtype=torch.long)
        return {'candidates':fused[:-1],'terminal':fused[-1],
                'candidate_mask':batch['candidate_mask'],'feasible_mask':batch['feasible_mask'],
                'rel_logits':rel_logits,'rel_targets':rel_targets}

    def proposer_logits(self, encoded, prefix, step=0):
        candidates=encoded['candidates']; k=len(candidates)
        state=torch.tanh(self.initial(encoded['terminal']))
        for index in prefix:
            if not 0 <= int(index) < k:raise ValueError('Invalid proposer prefix')
            state=self.prefix_update(candidates[int(index)].unsqueeze(0),state.unsqueeze(0))[0]
        query=self.pointer(state+self.step_embedding.weight[min(step,self.config.max_shortlist-1)])
        items=torch.cat([candidates,encoded['terminal'].unsqueeze(0)],0)
        logits=items @ query / (self.config.d_model ** .5)
        logits=torch.cat([logits[:k],(logits[k]+self.stop_head(state)[0]).unsqueeze(0)])
        valid=torch.cat([encoded['candidate_mask'].bool(),torch.ones(1,device=logits.device,dtype=torch.bool)])
        if prefix:
            valid=valid.clone();valid[torch.tensor(prefix,device=logits.device)]=False
        return logits.masked_fill(~valid,torch.finfo(logits.dtype).min)

    def arbiter_logits(self, encoded):
        state=torch.tanh(self.initial(encoded['terminal']))
        logits=encoded['candidates'] @ self.pointer(state) / (self.config.d_model ** .5)
        terminal=self.stop_head(encoded['terminal'])[0]
        logits=torch.cat([logits,terminal.unsqueeze(0)])
        valid=torch.cat([encoded['candidate_mask'].bool(),torch.ones(1,device=logits.device,dtype=torch.bool)])
        return logits.masked_fill(~valid,torch.finfo(logits.dtype).min)

    def forward(self,batch):
        return self.encode(batch)

    def architecture_config(self):
        return asdict(self.config)
