from __future__ import annotations

from math import sqrt

import torch
from torch_geometric.nn import knn, knn_graph
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter


def _cross(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Cross product with broadcasting across the node dimension."""
    return torch.stack(
        (
            lhs[:, 1] * rhs[:, 2] - lhs[:, 2] * rhs[:, 1],
            lhs[:, 2] * rhs[:, 0] - lhs[:, 0] * rhs[:, 2],
            lhs[:, 0] * rhs[:, 1] - lhs[:, 1] * rhs[:, 0],
        ),
        dim=1,
    )


def surface_to_edges(
    positions: torch.Tensor,
    normals: torch.Tensor,
    up_vector: torch.Tensor | None = None,
    neighbors: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a local tangent-plane KNN graph and return its 2D directions."""
    if up_vector is None:
        up_vector = torch.tensor([[0.0, 1.0, 0.0]], device=positions.device)
    edge_index = knn_graph(positions, int(neighbors), loop=True)
    aligned = (normals[edge_index[1]] * normals[edge_index[0]]).sum(dim=1) > 0
    edge_index = edge_index[:, aligned]

    surface_normals = normals[edge_index[0]]
    surface_normals = surface_normals / surface_normals.norm(dim=1, keepdim=True)
    x_axis = _cross(up_vector, surface_normals)
    x_axis = x_axis / x_axis.norm(dim=1, keepdim=True)
    y_axis = _cross(surface_normals, x_axis)
    y_axis = y_axis / y_axis.norm(dim=1, keepdim=True)

    directions = positions[edge_index[1]] - positions[edge_index[0]]
    original = directions.clone()
    directions[:, 0] = (original * x_axis).sum(dim=1)
    directions[:, 1] = (original * y_axis).sum(dim=1)
    return edge_index, directions[:, :2]


def _interpolate_selections(
    edge_index: torch.Tensor,
    directions: torch.Tensor,
    vectors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction_norm = directions.norm(dim=1, keepdim=True)
    unit_directions = directions / direction_norm
    values = unit_directions @ vectors
    best = values.argmax(dim=1, keepdim=True)
    best_value = values.take_along_dim(best, dim=1)
    lower = values.take_along_dim((best - 1) % 8, dim=1)
    upper = values.take_along_dim((best + 1) % 8, dim=1)
    neighbors = torch.cat((lower, upper), dim=1)
    second_value = neighbors.max(dim=1).values
    second = neighbors.argmax(dim=1)

    best_value = torch.minimum(best_value[:, 0], torch.tensor(1.0, device=directions.device))
    best_angle = torch.arccos(best_value)
    second_angle = torch.arccos(second_value)
    angle = best_angle / (second_angle + best_angle)
    angle[second == 0] *= -1
    angle = torch.nan_to_num(angle)

    selections = best[:, 0] + 1
    same_node = edge_index[0] == edge_index[1]
    selections[same_node] = 0
    angle[same_node] = 0
    interpolations = 1.0 - angle.abs()

    positive = angle > 1e-2
    positive_edges = edge_index[:, positive]
    positive_selections = selections[positive] + 1
    positive_selections[positive_selections > 8] = 1
    positive_interpolations = angle[positive]

    negative = angle < -1e-2
    negative_edges = edge_index[:, negative]
    negative_selections = selections[negative] - 1
    negative_selections[negative_selections < 1] = 8
    negative_interpolations = angle[negative].abs()

    return (
        torch.cat((edge_index, positive_edges, negative_edges), dim=1),
        torch.cat((selections, positive_selections, negative_selections), dim=0),
        torch.cat(
            (interpolations, positive_interpolations, negative_interpolations), dim=0
        ),
    )


def _normalize_edges(
    edge_index: torch.Tensor,
    selections: torch.Tensor,
    interpolations: torch.Tensor,
) -> torch.Tensor:
    node_count = int(edge_index.max()) + 1
    selection_count = int(selections.max()) + 1
    total_weight = torch.zeros(
        (node_count, selection_count), dtype=torch.float32, device=edge_index.device
    )
    nodes = edge_index[0]
    total_weight.index_put_((nodes, selections), interpolations, accumulate=True)
    norms = total_weight[nodes, selections].clamp_min(1e-6)
    return interpolations / norms


def edges_to_selections(
    edge_index: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map tangent directions to the interpolated 3x3 SelectionConv kernel."""
    diagonal = sqrt(2.0) / 2.0
    vectors = torch.tensor(
        [
            [1.0, 0.0],
            [diagonal, diagonal],
            [0.0, 1.0],
            [-diagonal, diagonal],
            [-1.0, 0.0],
            [-diagonal, -diagonal],
            [0.0, -1.0],
            [diagonal, -diagonal],
        ],
        dtype=torch.float32,
        device=directions.device,
    ).transpose(0, 1)
    edge_index, selections, interpolations = _interpolate_selections(
        edge_index, directions, vectors
    )
    interpolations = _normalize_edges(edge_index, selections, interpolations)
    return edge_index, selections, interpolations


def make_surface_clusters(
    positions: torch.Tensor,
    normals: torch.Tensor,
    edge_index: torch.Tensor,
    selections: torch.Tensor,
    interpolations: torch.Tensor,
    ratio: float = 0.25,
    up_vector: torch.Tensor | None = None,
    depth: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Construct pooled graph levels while preserving SelectionConv neighborhoods."""
    clusters = []
    edge_indexes = [edge_index.clone().to(device)]
    selections_list = [selections.clone().to(device)]
    interpolations_list = [interpolations.clone().to(device)]

    for _ in range(int(depth)):
        next_size = int(len(positions) * float(ratio))
        sampled = torch.multinomial(
            torch.ones(len(positions), device=positions.device), next_size
        )
        centroids = positions[sampled]
        cluster = knn(centroids, positions, 1)[1]
        cluster, _ = consecutive_cluster(cluster)
        positions = scatter(positions, cluster, dim=0, reduce="mean")
        normals = scatter(normals, cluster, dim=0, reduce="mean")
        normals = normals / normals.norm(dim=1, keepdim=True)

        edge_index, directions = surface_to_edges(
            positions, normals, up_vector=up_vector, neighbors=16
        )
        edge_index, selections, interpolations = edges_to_selections(
            edge_index, directions
        )
        clusters.append(cluster.clone().to(device))
        edge_indexes.append(edge_index.clone().to(device))
        selections_list.append(selections.clone().to(device))
        interpolations_list.append(interpolations.clone().to(device))

    return clusters, edge_indexes, selections_list, interpolations_list
