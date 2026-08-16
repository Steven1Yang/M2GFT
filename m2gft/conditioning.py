from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .base.image_vgg import ImageVGGEncoder


PYRAMID_CHANNELS = {"r11": 64, "r21": 128, "r31": 256, "r41": 512}


def gram_matrix(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(feature.shape)}")
    batch, channels, height, width = feature.shape
    flat = feature.reshape(batch, channels, height * width)
    return torch.bmm(flat, flat.transpose(1, 2)) / float(channels * height * width)


def feature_mean_std(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = feature.flatten(2)
    return flat.mean(dim=2), flat.var(dim=2, unbiased=False).add(1e-8).sqrt()


class FrozenImagePyramid(nn.Module):
    """Frozen 2D VGG used to construct the style feature pyramid."""

    def __init__(self, weights_path: str | Path):
        super().__init__()
        self.encoder = ImageVGGEncoder()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Images must be [B,3,H,W], got {tuple(images.shape)}")
        features = self.encoder(images.float().clamp(0.0, 1.0))
        return {level: features[level] for level in PYRAMID_CHANNELS}
