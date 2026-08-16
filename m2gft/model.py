from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .base.graph_vgg import GraphVGGEncoder
from .base.image_vgg import ImageVGGEncoder
from .base.r41_transform import R41GraphStyleTransform
from .conditioning import FrozenImagePyramid
from .pyramid.decoder import PyramidGraphDecoder


class M2GFTStylizer(nn.Module):
    """Four-level graph feature transformation for Gaussian splats."""

    levels = ("r11", "r21", "r31", "r41")

    def __init__(
        self,
        encoder_weights: str | Path,
        decoder_weights: str | Path,
        r41_weights: str | Path,
        local_blocks: int = 2,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        image_encoder = ImageVGGEncoder()
        image_encoder.load_state_dict(
            torch.load(encoder_weights, map_location="cpu", weights_only=True)
        )
        self.graph_encoder = GraphVGGEncoder(padding_mode=padding_mode)
        with torch.no_grad():
            self.graph_encoder.copy_weights(image_encoder)
        del image_encoder

        self.style_encoder = FrozenImagePyramid(encoder_weights)
        self.r41 = R41GraphStyleTransform(r41_weights)
        self.decoder = PyramidGraphDecoder(
            decoder_weights,
            local_blocks=local_blocks,
            padding_mode=padding_mode,
        )
        for module in (self.graph_encoder, self.style_encoder, self.r41):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.graph_encoder.eval()
        self.style_encoder.eval()
        self.r41.eval()
        return self

    def set_last_decoder_trainable(self, enabled: bool) -> None:
        for parameter in self.decoder.last_backbone_parameters():
            parameter.requires_grad_(bool(enabled))

    def optimizer_groups(
        self,
        lr: float,
        decoder_lr_scale: float = 0.05,
        include_decoder_last: bool = True,
    ):
        groups = [
            {
                "name": "pyramid_generator",
                "params": list(self.decoder.blocks.parameters()),
                "lr": float(lr),
            }
        ]
        if include_decoder_last:
            groups.append(
                {
                    "name": "decoder_last",
                    "params": list(self.decoder.last_backbone_parameters()),
                    "lr": float(lr) * float(decoder_lr_scale),
                }
            )
        return groups

    def encode_graph(self, graph) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features = self.graph_encoder(graph)
        return {level: features[level].detach() for level in self.levels}

    def encode_style(self, style_image: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features = self.style_encoder(style_image)
        return {level: features[level].detach() for level in self.levels}

    def forward(self, graph, style_image: torch.Tensor, return_details: bool = False):
        content = self.encode_graph(graph)
        style = self.encode_style(style_image)
        with torch.no_grad():
            coarse = self.r41(content["r41"], style["r41"], graph)
        raw_rgb, trace = self.decoder(
            coarse,
            content,
            style,
            graph,
            use_pyramid_generator=True,
            return_trace=True,
        )
        rgb = raw_rgb.clamp(0.0, 1.0)
        if not return_details:
            return rgb
        with torch.no_grad():
            reference_raw = self.decoder(
                coarse,
                content,
                style,
                graph,
                use_pyramid_generator=False,
            )
            reference_rgb = reference_raw.clamp(0.0, 1.0)
        return {
            "rgb": rgb,
            "raw_rgb": raw_rgb,
            "reference_rgb": reference_rgb,
            "reference_raw": reference_raw,
            "content": content,
            "style": style,
            "coarse": coarse,
            "trace": trace,
        }
