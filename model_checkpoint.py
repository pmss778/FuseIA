"""Complete dual-path checkpoints without duplicating frozen base-model weights."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random

import torch

from policy_schema import FEATURE_SCHEMA
from semantic_structure_model import ModelConfig, SemanticStructureModel


def save_checkpoint(path, model, optimizer=None, scheduler=None, progress=None,
                    dataset_summary=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.hf.save_pretrained(path / 'adapter', safe_serialization=True)
    if model.backbone.tokenizer is not None:
        model.backbone.tokenizer.save_pretrained(path / 'tokenizer')
    network = {key: value.detach().cpu() for key,value in model.state_dict().items()
               if not key.startswith('backbone.hf.')}
    temporary = path / 'network.tmp.pt'
    torch.save(network, temporary)
    temporary.replace(path / 'network.pt')
    metadata = {'model_config': asdict(model.config), 'feature_schema': FEATURE_SCHEMA,
                'progress': progress or {}, 'dataset': dataset_summary or {}}
    (path / 'checkpoint.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    if optimizer is not None:
        training = {'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'progress': progress or {}}
        torch.save(training, path / 'training.tmp.pt')
        (path / 'training.tmp.pt').replace(path / 'training.pt')


def restore_model_weights(path, model):
    """Load all LoRA and side-network parameters into the same architecture."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    path = Path(path)
    metadata = json.loads((path / 'checkpoint.json').read_text())
    if metadata['feature_schema'] != FEATURE_SCHEMA:
        raise ValueError('Checkpoint feature/relation schema differs from implementation')
    if metadata['model_config']['role'] != model.config.role:
        raise ValueError('Checkpoint belongs to another role')
    adapter = load_file(str(path / 'adapter/adapter_model.safetensors'))
    set_peft_model_state_dict(model.backbone.hf, adapter)
    network = torch.load(path / 'network.pt', map_location='cpu', weights_only=True)
    expected = {k for k in model.state_dict() if not k.startswith('backbone.hf.')}
    if set(network) != expected:
        raise ValueError('Side network checkpoint is incomplete or structurally different')
    model.load_state_dict(network, strict=False)
    return metadata


def load_checkpoint(path, device='cuda', backbone_name=None):
    path = Path(path)
    metadata = json.loads((path / 'checkpoint.json').read_text())
    config = ModelConfig(**metadata['model_config'])
    if backbone_name is not None:
        config.backbone_name = str(backbone_name)
    model = SemanticStructureModel(config)
    restore_model_weights(path, model)
    return model.to(device), metadata


def restore_training(path, optimizer, scheduler=None):
    training = torch.load(Path(path) / 'training.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(training['optimizer'])
    if scheduler is not None and training.get('scheduler') is not None:
        scheduler.load_state_dict(training['scheduler'])
    torch.set_rng_state(training['torch_rng'])
    random.setstate(training['python_rng'])
    if training.get('cuda_rng') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(training['cuda_rng'])
    return training['progress']
