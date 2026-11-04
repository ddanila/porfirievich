"""Load explicit experimental bias patches without changing checkpoint files."""

import hashlib
import json
from pathlib import Path

import torch


def apply_style_patch(model, model_path: Path, patch_path: Path) -> str:
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    if patch.get("format") != "porfirievich-mlp-bias-v1":
        raise ValueError("Unsupported style patch format")
    layer = patch.get("layer")
    if type(layer) is not int or not 0 <= layer < len(model.transformer.h):
        raise ValueError("Style patch layer is outside this model")
    bias = model.transformer.h[layer].mlp.c_proj.bias
    delta = torch.tensor(patch["delta"], dtype=bias.dtype)
    if delta.shape != bias.shape or not torch.isfinite(delta).all():
        raise ValueError("Style patch must contain one finite value per output bias")
    digest = hashlib.sha256()
    with (model_path / "pytorch_model.bin").open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != patch.get("base_checkpoint_sha256"):
        raise ValueError("Style patch was created for a different checkpoint")
    with torch.no_grad():
        bias.add_(delta.to(bias.device))
    return str(patch.get("name", patch_path.name))
