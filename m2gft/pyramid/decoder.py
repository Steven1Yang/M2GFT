from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.image_vgg import ImageVGGDecoder
from ..base.r41_transform import graph_level
from ..ops.pooling import unpool_cluster
from ..ops.selection_conv import SelectionConv
from .block import PyramidStyleBlock


class PyramidGraphDecoder(nn.Module):
    """Decode r41 while regenerating r31, r21, and r11 graph features."""

    def __init__(
        self,
        decoder_weights: str | Path,
        local_blocks: int = 2,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        channels = {
            11: (512, 256),
            12: (256, 256),
            13: (256, 256),
            14: (256, 256),
            15: (256, 128),
            16: (128, 128),
            17: (128, 64),
            18: (64, 64),
            19: (64, 3),
        }
        for index, (in_channels, out_channels) in channels.items():
            setattr(
                self,
                f"conv{index}",
                SelectionConv(
                    in_channels,
                    out_channels,
                    3,
                    padding_mode=padding_mode,
                ),
            )
        self.blocks = nn.ModuleDict(
            {
                "r31": PyramidStyleBlock(
                    256, 2, 4, local_blocks=local_blocks, padding_mode=padding_mode
                ),
                "r21": PyramidStyleBlock(
                    128, 1, 6, local_blocks=local_blocks, padding_mode=padding_mode
                ),
                "r11": PyramidStyleBlock(
                    64, 0, 8, local_blocks=local_blocks, padding_mode=padding_mode
                ),
            }
        )

        source = ImageVGGDecoder()
        source.load_state_dict(
            torch.load(decoder_weights, map_location="cpu", weights_only=True)
        )
        with torch.no_grad():
            for index in range(11, 20):
                graph_layer = getattr(self, f"conv{index}")
                image_layer = getattr(source, f"conv{index}")
                graph_layer.copy_weights(image_layer.weight, image_layer.bias)
        del source
        for parameter in self.backbone_parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _conv(layer, value, graph, level: int):
        edge_index, selections, interps = graph_level(graph, level)
        return layer(value, edge_index, selections, interps)

    def forward(
        self,
        coarse: torch.Tensor,
        content: dict[str, torch.Tensor],
        style: dict[str, torch.Tensor],
        graph,
        use_pyramid_generator: bool = True,
        return_trace: bool = False,
    ):
        trace = {}
        out = F.relu(self._conv(self.conv11, coarse, graph, 3))
        out = unpool_cluster(out, graph.clusters[2])
        if use_pyramid_generator:
            out = self.blocks["r31"](out, content["r31"], style["r31"], graph)
        trace["r31"] = out
        out = F.relu(self._conv(self.conv12, out, graph, 2))
        out = F.relu(self._conv(self.conv13, out, graph, 2))
        out = F.relu(self._conv(self.conv14, out, graph, 2))
        out = F.relu(self._conv(self.conv15, out, graph, 2))

        out = unpool_cluster(out, graph.clusters[1])
        if use_pyramid_generator:
            out = self.blocks["r21"](out, content["r21"], style["r21"], graph)
        trace["r21"] = out
        out = F.relu(self._conv(self.conv16, out, graph, 1))
        out = F.relu(self._conv(self.conv17, out, graph, 1))

        out = unpool_cluster(out, graph.clusters[0])
        if use_pyramid_generator:
            out = self.blocks["r11"](out, content["r11"], style["r11"], graph)
        trace["r11"] = out
        out = F.relu(self._conv(self.conv18, out, graph, 0))
        rgb = self._conv(self.conv19, out, graph, 0)
        return (rgb, trace) if return_trace else rgb

    def backbone_parameters(self):
        for index in range(11, 20):
            yield from getattr(self, f"conv{index}").parameters()

    def last_backbone_parameters(self):
        for index in (18, 19):
            yield from getattr(self, f"conv{index}").parameters()
