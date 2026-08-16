from __future__ import annotations

import torch
import torch.nn as nn

from ..ops.pooling import max_pool_cluster
from ..ops.selection_conv import SelectionConv


class GraphVGGEncoder(nn.Module):
    """SelectionConv counterpart of the R41 image feature pyramid."""

    def __init__(self, padding_mode: str = "reflect"):
        super().__init__()
        self.conv1 = SelectionConv(3, 3, 1, padding_mode=padding_mode)
        self.conv2 = SelectionConv(3, 64, 3, padding_mode=padding_mode)
        self.conv3 = SelectionConv(64, 64, 3, padding_mode=padding_mode)
        self.conv4 = SelectionConv(64, 128, 3, padding_mode=padding_mode)
        self.conv5 = SelectionConv(128, 128, 3, padding_mode=padding_mode)
        self.conv6 = SelectionConv(128, 256, 3, padding_mode=padding_mode)
        self.conv7 = SelectionConv(256, 256, 3, padding_mode=padding_mode)
        self.conv8 = SelectionConv(256, 256, 3, padding_mode=padding_mode)
        self.conv9 = SelectionConv(256, 256, 3, padding_mode=padding_mode)
        self.conv10 = SelectionConv(256, 512, 3, padding_mode=padding_mode)

    def copy_weights(self, image_encoder: nn.Module) -> None:
        for index in range(1, 11):
            target = getattr(self, f"conv{index}")
            source = getattr(image_encoder, f"conv{index}")
            target.copy_weights(source.weight, source.bias)

    @staticmethod
    def _level(graph, index: int):
        interpolations = graph.interps_list[index] if hasattr(graph, "interps_list") else None
        return graph.edge_indexes[index], graph.selections_list[index], interpolations

    def forward(self, graph) -> dict[str, torch.Tensor]:
        output = {}
        edges, selections, interps = self._level(graph, 0)
        value = self.conv1(graph.x, edges, selections, interps)
        output["r11"] = torch.relu(self.conv2(value, edges, selections, interps))
        output["r12"] = torch.relu(self.conv3(output["r11"], edges, selections, interps))
        output["p1"] = max_pool_cluster(output["r12"], graph.clusters[0])

        edges, selections, interps = self._level(graph, 1)
        output["r21"] = torch.relu(self.conv4(output["p1"], edges, selections, interps))
        output["r22"] = torch.relu(self.conv5(output["r21"], edges, selections, interps))
        output["p2"] = max_pool_cluster(output["r22"], graph.clusters[1])

        edges, selections, interps = self._level(graph, 2)
        output["r31"] = torch.relu(self.conv6(output["p2"], edges, selections, interps))
        output["r32"] = torch.relu(self.conv7(output["r31"], edges, selections, interps))
        output["r33"] = torch.relu(self.conv8(output["r32"], edges, selections, interps))
        output["r34"] = torch.relu(self.conv9(output["r33"], edges, selections, interps))
        output["p3"] = max_pool_cluster(output["r34"], graph.clusters[2])

        edges, selections, interps = self._level(graph, 3)
        output["r41"] = torch.relu(self.conv10(output["p3"], edges, selections, interps))
        return output
