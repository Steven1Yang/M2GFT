from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.r41_transform import graph_level
from ..ops.selection_conv import SelectionConv


class PyramidStyleBlock(nn.Module):
    """Generate one graph-pyramid level from content, top-down, and style features."""

    def __init__(
        self,
        channels: int,
        graph_level_index: int,
        token_grid: int,
        attention_dim: int = 32,
        local_blocks: int = 2,
        padding_mode: str = "replicate",
        attention_chunk: int = 32768,
    ):
        super().__init__()
        self.channels = int(channels)
        self.graph_level_index = int(graph_level_index)
        self.token_grid = int(token_grid)
        self.attention_dim = int(min(attention_dim, channels))
        self.attention_chunk = int(attention_chunk)
        token_count = self.token_grid * self.token_grid

        self.query_norm = nn.LayerNorm(2 * self.channels)
        self.style_norm = nn.LayerNorm(self.channels)
        self.query = nn.Linear(2 * self.channels, self.attention_dim)
        self.key = nn.Linear(self.channels, self.attention_dim)
        self.value = nn.Linear(self.channels, self.channels)
        self.key_position = nn.Parameter(torch.zeros(token_count, self.attention_dim))
        self.value_position = nn.Parameter(torch.zeros(token_count, self.channels))
        nn.init.trunc_normal_(self.key_position, std=0.02)
        nn.init.trunc_normal_(self.value_position, std=0.02)

        self.input_projection = nn.Linear(4 * self.channels, self.channels)
        self.local_convs = nn.ModuleList(
            [
                SelectionConv(
                    self.channels,
                    self.channels,
                    3,
                    padding_mode=padding_mode,
                )
                for _ in range(int(local_blocks))
            ]
        )
        self.local_norms = nn.ModuleList(
            [nn.LayerNorm(self.channels) for _ in range(int(local_blocks))]
        )
        self.output_projection = nn.Linear(2 * self.channels, self.channels)

        with torch.no_grad():
            self.input_projection.weight.zero_()
            self.input_projection.bias.zero_()
            self.input_projection.weight[:, : self.channels].copy_(
                torch.eye(self.channels)
            )
            self.output_projection.weight.zero_()
            self.output_projection.bias.zero_()
            self.output_projection.weight[:, : self.channels].copy_(
                torch.eye(self.channels)
            )

    def _style_tokens(self, style: torch.Tensor) -> torch.Tensor:
        if style.ndim != 4 or style.shape[0] != 1:
            raise ValueError(f"Expected one style feature map, got {tuple(style.shape)}")
        pooled = F.adaptive_avg_pool2d(style, (self.token_grid, self.token_grid))
        return pooled.flatten(2).transpose(1, 2).squeeze(0)

    def _attend(
        self,
        decoder_feature: torch.Tensor,
        content: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.style_norm(self._style_tokens(style))
        keys = self.key(tokens) + self.key_position
        values = self.value(tokens) + self.value_position
        queries = self.query(
            self.query_norm(torch.cat([decoder_feature, content], dim=1))
        )
        scale = float(self.attention_dim) ** -0.5
        pieces = []
        for start in range(0, queries.shape[0], self.attention_chunk):
            query = queries[start : start + self.attention_chunk]
            weights = torch.softmax(query @ keys.transpose(0, 1) * scale, dim=1)
            pieces.append(weights @ values)
        return torch.cat(pieces, dim=0)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        content: torch.Tensor,
        style: torch.Tensor,
        graph,
    ) -> torch.Tensor:
        if decoder_feature.shape != content.shape:
            raise ValueError(
                f"Pyramid features must match, got {tuple(decoder_feature.shape)} "
                f"and {tuple(content.shape)}"
            )
        content_mean = content.mean(dim=0, keepdim=True)
        content_std = content.var(dim=0, unbiased=False, keepdim=True).add(1e-6).sqrt()
        style_mean = style.mean(dim=(0, 2, 3), keepdim=False).reshape(1, -1)
        style_std = (
            style.var(dim=(0, 2, 3), unbiased=False, keepdim=False)
            .add(1e-6)
            .sqrt()
            .reshape(1, -1)
        )
        normalized_content = (content - content_mean) / content_std
        adain_content = normalized_content * style_std + style_mean
        attended_style = self._attend(decoder_feature, content, style)
        generated = self.input_projection(
            torch.cat(
                [decoder_feature, normalized_content, adain_content, attended_style],
                dim=1,
            )
        )
        local = generated
        edge_index, selections, interps = graph_level(graph, self.graph_level_index)
        for conv, norm in zip(self.local_convs, self.local_norms):
            local = F.gelu(norm(conv(local, edge_index, selections, interps)))
        return self.output_projection(torch.cat([generated, local], dim=1))
