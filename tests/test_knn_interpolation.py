from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2gft.gaussians import GaussianCloud
from m2gft.graph import (
    SurfaceGraph,
    _knn_mapping,
    build_surface_graph,
    interpolate_node_values,
)


def make_graph(graph_positions: torch.Tensor, original_positions: torch.Tensor) -> SurfaceGraph:
    indices, weights = _knn_mapping(graph_positions, original_positions, neighbors=2)
    return SurfaceGraph(
        data=Data(x=torch.zeros(len(graph_positions), 3)),
        positions=graph_positions,
        original_to_nodes=indices,
        original_to_weights=weights,
        source="unit-test",
    )


def test_knn_inverse_distance_interpolation_and_backward():
    graph_positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    original_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    graph = make_graph(graph_positions, original_positions)
    values = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True)
    interpolated = interpolate_node_values(values, graph, chunk_size=2)
    assert interpolated.shape == (3, 3)
    assert torch.allclose(interpolated[0], values[0], atol=1e-6)
    assert torch.allclose(interpolated[1], torch.tensor([0.5, 0.0, 0.5]), atol=1e-6)
    assert torch.allclose(interpolated[2], values[1], atol=1e-6)
    interpolated.sum().backward()
    assert values.grad is not None and torch.all(values.grad > 0)


def test_knn_rgb_interpolation_stays_in_convex_hull():
    graph_positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    original_positions = torch.tensor([[0.25, 0.0, 0.0], [1.25, 0.0, 0.0]])
    graph = make_graph(graph_positions, original_positions)
    colors = torch.tensor([[0.1, 0.4, 0.9], [0.8, 0.6, 0.2]])
    interpolated = interpolate_node_values(colors, graph)
    assert interpolated.min() >= colors.min() - 1e-6
    assert interpolated.max() <= colors.max() + 1e-6


def test_surface_graph_builder_creates_four_valid_levels():
    generator = torch.Generator().manual_seed(7)
    count = 256
    cloud = GaussianCloud(
        means=torch.randn(count, 3, generator=generator),
        scales=torch.full((count, 3), 0.01),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        colors=torch.rand(count, 3, generator=generator),
        opacities=torch.ones(count),
        source="synthetic-test",
    )
    graph = build_surface_graph(
        cloud,
        max_nodes=count,
        normal_neighbors=8,
        edge_neighbors=9,
        seed=7,
        build_device="cpu",
        mapping_neighbors=2,
    )
    assert graph.level_sizes == (256, 64, 16, 4)
    assert graph.original_to_nodes.shape == (count, 2)
    assert torch.allclose(graph.original_to_weights.sum(dim=1), torch.ones(count))
    for edges, selections, interps in zip(
        graph.data.edge_indexes,
        graph.data.selections_list,
        graph.data.interps_list,
    ):
        assert edges.shape[1] == selections.shape[0] == interps.shape[0]
        assert torch.isfinite(interps).all()


if __name__ == "__main__":
    test_knn_inverse_distance_interpolation_and_backward()
    test_knn_rgb_interpolation_stays_in_convex_hull()
    test_surface_graph_builder_creates_four_valid_levels()
    print("KNN interpolation tests passed")
