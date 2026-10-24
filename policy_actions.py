"""Shared categorical sequence protocol for SFT, inference and GRPO."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def action_distribution(model, encoded, prefix=None, step=0, temperature=1.0,
                        feasibility_mask=False):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    if model.config.role == 'proposer':
        logits = model.proposer_logits(encoded, prefix or [], step)
    else:
        logits = model.arbiter_logits(encoded)
    if feasibility_mask:
        valid = torch.cat([encoded['feasible_mask'].bool(),
                           torch.ones(1, device=logits.device, dtype=torch.bool)])
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    return torch.distributions.Categorical(logits=logits.float()/temperature)


def evaluate_actions(model, encoded, actions, temperature=1.0,
                     feasibility_mask=False):
    """Exactly the same prefix, mask and temperature as actual sampling."""
    k = len(encoded['candidates'])
    prefix, logprobs, entropy, logits = [], [], [], []
    for step, action in enumerate(actions):
        action = int(action)
        if not 0 <= action <= k:
            raise ValueError('Action outside candidate/terminal space')
        if action in prefix:
            raise ValueError('Duplicate candidate in sequence')
        if model.config.role == 'arbiter' and step:
            raise ValueError('Arbiter must make exactly one decision')
        if step >= model.config.max_shortlist:
            raise ValueError('Proposer sequence exceeds cap')
        dist = action_distribution(model, encoded, prefix, step, temperature, feasibility_mask)
        logits.append(dist.logits)
        index = torch.tensor(action, device=dist.logits.device)
        logprobs.append(dist.log_prob(index)); entropy.append(dist.entropy())
        if action == k:
            if step != len(actions)-1:
                raise ValueError('Actions after END/STOP')
            break
        prefix.append(action)
    if not logprobs:
        raise ValueError('Empty action record; empty shortlist must contain END')
    if model.config.role == 'proposer' and len(prefix) < model.config.max_shortlist and int(actions[-1]) != k:
        raise ValueError('Short shortlist has no END')
    return {'logprobs':torch.stack(logprobs), 'entropy':torch.stack(entropy),
            'logits':torch.stack(logits), 'sequence_logprob':torch.stack(logprobs).sum()}


def select_actions(model, encoded, greedy=False, temperature=0.8,
                   feasibility_mask=False):
    k = len(encoded['candidates'])
    prefix, actions, logprobs = [], [], []
    cap = model.config.max_shortlist if model.config.role == 'proposer' else 1
    for step in range(cap):
        dist = action_distribution(model, encoded, prefix, step, temperature, feasibility_mask)
        action = dist.logits.argmax() if greedy else dist.sample()
        index = int(action)
        actions.append(index); logprobs.append(dist.log_prob(action))
        if index == k:
            break
        prefix.append(index)
    return {'actions':actions, 'candidate_indices':prefix,
            'terminated':actions[-1]==k, 'logprob':torch.stack(logprobs).sum(),
            'temperature':temperature, 'feasibility_mask':feasibility_mask}


def symmetric_action_kl(first, second):
    if first['logits'].shape != second['logits'].shape:
        raise ValueError('Paired views have different candidate/action positions')
    p = F.log_softmax(first['logits'].float(), -1)
    q = F.log_softmax(second['logits'].float(), -1)
    return (0.5*((p.exp()*(p-q)).sum(-1)+(q.exp()*(q-p)).sum(-1))).mean()
