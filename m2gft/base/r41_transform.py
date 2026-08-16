from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.selection_conv import SelectionConv


def graph_level(graph, level: int):
    interps = graph.interps_list[level] if hasattr(graph, "interps_list") else None
    return graph.edge_indexes[level], graph.selections_list[level], interps


class R41GraphStyleTransform(nn.Module):
    """R41 transform with graph-content and image-style paths."""

    def __init__(self, weights_path: str | Path, matrix_size: int = 32):
        super().__init__()
        self.matrix_size = int(matrix_size)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.content_convs = nn.ModuleList(
            [
                SelectionConv(512, 256, 3, padding_mode="zeros"),
                SelectionConv(256, 128, 3, padding_mode="zeros"),
                SelectionConv(128, self.matrix_size, 3, padding_mode="zeros"),
            ]
        )
        self.content_fc = nn.Linear(
            self.matrix_size * self.matrix_size,
            self.matrix_size * self.matrix_size,
        )
        self.style_convs = nn.ModuleList(
            [
                nn.Conv2d(512, 256, 3, padding=1),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.Conv2d(128, self.matrix_size, 3, padding=1),
            ]
        )
        self.style_fc = nn.Linear(
            self.matrix_size * self.matrix_size,
            self.matrix_size * self.matrix_size,
        )
        self.compress = SelectionConv(512, self.matrix_size, 1)
        self.unzip = SelectionConv(self.matrix_size, 512, 1)

        with torch.no_grad():
            for index, layer in enumerate(self.content_convs):
                source_index = 2 * index
                layer.copy_weights(
                    nn.Parameter(state[f"cnet.convs.{source_index}.weight"].clone()),
                    nn.Parameter(state[f"cnet.convs.{source_index}.bias"].clone()),
                )
            for index, layer in enumerate(self.style_convs):
                source_index = 2 * index
                layer.weight.copy_(state[f"snet.convs.{source_index}.weight"])
                layer.bias.copy_(state[f"snet.convs.{source_index}.bias"])
            self.content_fc.weight.copy_(state["cnet.fc.weight"])
            self.content_fc.bias.copy_(state["cnet.fc.bias"])
            self.style_fc.weight.copy_(state["snet.fc.weight"])
            self.style_fc.bias.copy_(state["snet.fc.bias"])
            self.compress.copy_weights(
                nn.Parameter(state["compress.weight"].clone()),
                nn.Parameter(state["compress.bias"].clone()),
            )
            self.unzip.copy_weights(
                nn.Parameter(state["unzip.weight"].clone()),
                nn.Parameter(state["unzip.bias"].clone()),
            )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _content_matrix(self, features: torch.Tensor, graph) -> torch.Tensor:
        edge_index, selections, interps = graph_level(graph, 3)
        out = features
        for index, layer in enumerate(self.content_convs):
            out = layer(out, edge_index, selections, interps)
            if index + 1 < len(self.content_convs):
                out = F.relu(out)
        covariance = out.transpose(0, 1) @ out / float(max(out.shape[0], 1))
        return self.content_fc(covariance.reshape(-1)).reshape(
            self.matrix_size,
            self.matrix_size,
        )

    def _style_matrix(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[0] != 1:
            raise ValueError(f"T41 expects one style feature batch, got {tuple(features.shape)}")
        out = features
        for index, layer in enumerate(self.style_convs):
            out = layer(out)
            if index + 1 < len(self.style_convs):
                out = F.relu(out)
        flat = out.flatten(2)
        covariance = torch.bmm(flat, flat.transpose(1, 2)) / float(max(flat.shape[2], 1))
        return self.style_fc(covariance.reshape(1, -1)).reshape(
            self.matrix_size,
            self.matrix_size,
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor, graph) -> torch.Tensor:
        edge_index, selections, interps = graph_level(graph, 3)
        content_centered = content - content.mean(dim=0, keepdim=True)
        style_mean = style.mean(dim=(0, 2, 3), keepdim=False).reshape(1, -1)
        style_centered = style - style_mean.reshape(1, -1, 1, 1)
        compressed = self.compress(content_centered, edge_index, selections, interps)
        transform = self._style_matrix(style_centered) @ self._content_matrix(
            content_centered,
            graph,
        )
        transformed = compressed @ transform.transpose(0, 1)
        return self.unzip(transformed, edge_index, selections, interps) + style_mean
