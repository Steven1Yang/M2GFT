from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2gft.model import M2GFTStylizer


def make_level_edges(nodes: int):
    node_ids = torch.arange(nodes).repeat_interleave(9)
    return (
        torch.stack([node_ids, node_ids]),
        torch.arange(9).repeat(nodes),
        torch.ones(nodes * 9),
    )


def make_graph():
    sizes = (256, 64, 16, 4)
    levels = [make_level_edges(nodes) for nodes in sizes]
    return Data(
        x=torch.rand(sizes[0], 3),
        clusters=[torch.arange(sizes[index]) // 4 for index in range(3)],
        edge_indexes=[item[0] for item in levels],
        selections_list=[item[1] for item in levels],
        interps_list=[item[2] for item in levels],
    )


def make_model():
    return M2GFTStylizer(
        ROOT / "weights/base/vgg_r41.pth",
        ROOT / "weights/base/dec_r41.pth",
        ROOT / "weights/base/r41.pth",
        local_blocks=1,
    )


def test_reference_initialization_without_additive_fusion():
    model = make_model().train()
    details = model(make_graph(), torch.rand(1, 3, 64, 64), return_details=True)
    assert details["rgb"].shape == (256, 3)
    assert torch.allclose(details["raw_rgb"], details["reference_raw"], atol=1e-6)
    assert all(not parameter.requires_grad for parameter in model.r41.parameters())
    assert all(not parameter.requires_grad for parameter in model.graph_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.decoder.backbone_parameters())


def test_only_pyramid_generator_receives_gradients():
    model = make_model().train()
    details = model(make_graph(), torch.rand(1, 3, 64, 64), return_details=True)
    details["raw_rgb"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.decoder.blocks.parameters())
    assert all(parameter.grad is None for parameter in model.r41.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.backbone_parameters())
    names = {group["name"] for group in model.optimizer_groups(1e-4)}
    assert names == {"pyramid_generator", "decoder_last"}


if __name__ == "__main__":
    test_reference_initialization_without_additive_fusion()
    test_only_pyramid_generator_receives_gradients()
    print("M2GFT model tests passed")
