"""Supervised categorical decisions, paired-view KL, and hidden relations."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from policy_actions import evaluate_actions, symmetric_action_kl


def relation_loss(encoded):
    labels = encoded['rel_targets']
    if labels.numel() == 0:
        return encoded['terminal'].sum() * 0
    return F.cross_entropy(encoded['rel_logits'].float(), labels)


def paired_sft_loss(model, original, anonymous, actions, rename_weight=0.1,
                    relation_weight=0.1):
    first = evaluate_actions(model, original, actions)
    second = evaluate_actions(model, anonymous, actions)
    action = -(first['logprobs'].mean() + second['logprobs'].mean()) / 2
    rename = symmetric_action_kl(first, second)
    relation = (relation_loss(original) + relation_loss(anonymous)) / 2
    total = action + rename_weight * rename + relation_weight * relation
    metrics = {'loss': float(total.detach()), 'action': float(action.detach()),
               'rename': float(rename.detach()), 'relation': float(relation.detach()),
               'decisions': len(actions),
               'teacher_forced_accuracy': float((first['logits'].argmax(-1) ==
                    torch.tensor(actions, device=first['logits'].device)).float().mean())}
    return total, metrics
