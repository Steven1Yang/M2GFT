from __future__ import annotations

import torch.nn as nn


ARCHITECTURE_ID = "m2gft_direct_rgb_v1"
RELEASE_FORMAT = "m2gft_release_v1"
LEARNED_PREFIX = "decoder.blocks."


def load_m2gft_checkpoint(model: nn.Module, checkpoint: dict) -> None:
    """Load either a full training checkpoint or the compact release format."""
    if checkpoint.get("architecture") != ARCHITECTURE_ID:
        raise ValueError(f"Unsupported M2GFT architecture: {checkpoint.get('architecture')!r}")
    if "trainable_state" not in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        return
    state = checkpoint["trainable_state"]
    if not state or any(not key.startswith(LEARNED_PREFIX) for key in state):
        raise ValueError("Release checkpoint must contain only pyramid-generator parameters")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected release checkpoint keys: {incompatible.unexpected_keys}")
    loaded = set(state)
    if any(key.startswith(LEARNED_PREFIX) and key not in loaded for key in model.state_dict()):
        raise RuntimeError("Release checkpoint is missing pyramid-generator parameters")
