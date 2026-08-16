from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_mean

from .gaussians import GaussianCloud
from .graph_ops import edges_to_selections, make_surface_clusters, surface_to_edges


@dataclass
class SurfaceGraph:
    data: Data
    positions: torch.Tensor
    original_to_nodes: torch.Tensor
    original_to_weights: torch.Tensor
    source: str

    def to(self, device: str | torch.device) -> "SurfaceGraph":
        return SurfaceGraph(
            data=self.data.to(device),
            positions=self.positions.to(device),
            original_to_nodes=self.original_to_nodes.to(device),
            original_to_weights=self.original_to_weights.to(device),
            source=self.source,
        )

    @property
    def mapping_neighbors(self) -> int:
        return int(self.original_to_nodes.shape[1])

    @property
    def original_to_node(self) -> torch.Tensor:
        """Legacy nearest-node view of the KNN mapping."""
        return self.original_to_nodes[:, 0].long()

    @property
    def level_sizes(self) -> tuple[int, int, int, int]:
        sizes = [int(self.data.x.shape[0])]
        for cluster in self.data.clusters:
            sizes.append(int(cluster.max().item()) + 1)
        return tuple(sizes)


def _sample_graph_nodes(cloud: GaussianCloud, max_nodes: int, seed: int) -> torch.Tensor:
    count = len(cloud)
    if max_nodes <= 0 or count <= max_nodes:
        return torch.arange(count, device=cloud.means.device)
    # Surface coverage remains stochastic but reproducible; scale and opacity bias the sample
    # toward visually important Gaussians without turning it into a top-k crop.
    weights = cloud.scales.prod(dim=1).pow(1.0 / 3.0) * cloud.opacities.clamp_min(1e-4)
    weights = weights.float().cpu().clamp_min(1e-12)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.multinomial(weights, int(max_nodes), replacement=False, generator=generator)
    return indices.to(cloud.means.device)


def _estimate_normals(positions: torch.Tensor, neighbors: int) -> torch.Tensor:
    if positions.shape[0] < 2:
        return torch.tensor([[0.0, 1.0, 0.0]], device=positions.device).expand_as(positions)
    k = min(max(2, int(neighbors)), int(positions.shape[0]))
    edges = knn_graph(positions, k=k, loop=True)
    # torch_geometric knn_graph stores neighbor -> target for source_to_target flow.
    local_mean = scatter_mean(positions[edges[0]], edges[1], dim=0, dim_size=positions.shape[0])
    normals = positions - local_mean
    fallback = positions - positions.mean(dim=0, keepdim=True)
    weak = normals.norm(dim=1, keepdim=True) < 1e-8
    normals = torch.where(weak, fallback, normals)
    normals = F.normalize(normals, dim=1, eps=1e-8)
    weak = normals.norm(dim=1, keepdim=True) < 0.5
    fixed = torch.zeros_like(normals)
    fixed[:, 1] = 1.0
    return torch.where(weak, fixed, normals)


def _knn_mapping(
    graph_positions: torch.Tensor,
    original_positions: torch.Tensor,
    neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K graph-node indices and normalized inverse-distance weights per Gaussian."""
    if neighbors <= 0:
        raise ValueError(f"mapping neighbors must be positive, got {neighbors}")
    neighbors = min(int(neighbors), int(graph_positions.shape[0]))
    tree = cKDTree(graph_positions.detach().float().cpu().numpy())
    distances, indices = tree.query(
        original_positions.detach().float().cpu().numpy(),
        k=neighbors,
        workers=-1,
    )
    if neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    distances = torch.from_numpy(distances.astype("float32", copy=False))
    # Graphs in this project are far below the signed int32 node limit. Keeping cached
    # indices compact matters for Garden's 5.8 million original Gaussians.
    indices = torch.from_numpy(indices.astype("int32", copy=False))
    inverse = distances.add(1e-8).reciprocal()
    weights = inverse / inverse.sum(dim=1, keepdim=True)
    return indices, weights


def interpolate_node_values(
    node_values: torch.Tensor,
    graph: SurfaceGraph,
    chunk_size: int = 1_000_000,
) -> torch.Tensor:
    """Differentiably interpolate graph-node values onto every original Gaussian."""
    if node_values.ndim != 2 or node_values.shape[0] != graph.data.x.shape[0]:
        raise ValueError(
            f"Expected node values [{graph.data.x.shape[0]},C], got {tuple(node_values.shape)}"
        )
    if graph.original_to_nodes.shape != graph.original_to_weights.shape:
        raise ValueError("KNN mapping indices and weights must have identical shapes")
    chunks = []
    total = graph.original_to_nodes.shape[0]
    for start in range(0, total, int(chunk_size)):
        stop = min(start + int(chunk_size), total)
        indices = graph.original_to_nodes[start:stop].long()
        weights = graph.original_to_weights[start:stop].to(dtype=node_values.dtype)
        gathered = node_values[indices]
        chunks.append((gathered * weights.unsqueeze(-1)).sum(dim=1))
    return torch.cat(chunks, dim=0)


def surface_graph_cache_name(name: str, max_nodes: int, seed: int, mapping_neighbors: int) -> str:
    return f"{name}_n{int(max_nodes)}_seed{int(seed)}_knn{int(mapping_neighbors)}.pt"


def build_surface_graph(
    cloud: GaussianCloud,
    max_nodes: int = 60000,
    normal_neighbors: int = 25,
    edge_neighbors: int = 16,
    ratio: float = 0.25,
    depth: int = 3,
    seed: int = 2964,
    build_device: str | torch.device = "cpu",
    mapping_neighbors: int = 4,
) -> SurfaceGraph:
    if depth != 3:
        raise ValueError("M2GFT's r11/r21/r31/r41 pyramid requires exactly three pooling levels")
    build_device = torch.device(build_device)
    cloud_on_device = cloud.to(build_device)
    indices = _sample_graph_nodes(cloud_on_device, max_nodes=max_nodes, seed=seed)
    positions = cloud_on_device.means[indices].contiguous()
    colors = cloud_on_device.colors[indices].contiguous()
    normals = _estimate_normals(positions, normal_neighbors)
    up_vector = F.normalize(torch.tensor([[1.0, 1.0, 1.0]], device=build_device), dim=1)

    edges, directions = surface_to_edges(
        positions, normals, up_vector=up_vector, neighbors=min(edge_neighbors, len(positions))
    )
    edges, selections, interps = edges_to_selections(edges, directions)
    with torch.random.fork_rng(devices=[build_device] if build_device.type == "cuda" else []):
        torch.manual_seed(int(seed))
        clusters, edge_indexes, selections_list, interps_list = make_surface_clusters(
            positions,
            normals,
            edges,
            selections,
            interps,
            ratio=float(ratio),
            up_vector=up_vector,
            depth=depth,
            device=build_device,
        )
    data = Data(
        x=colors,
        clusters=clusters,
        edge_indexes=edge_indexes,
        selections_list=selections_list,
        interps_list=interps_list,
    )
    original_to_nodes, original_to_weights = _knn_mapping(
        positions,
        cloud_on_device.means,
        neighbors=mapping_neighbors,
    )
    result = SurfaceGraph(
        data=data.cpu(),
        positions=positions.cpu(),
        original_to_nodes=original_to_nodes.cpu(),
        original_to_weights=original_to_weights.cpu(),
        source=cloud.source,
    )
    sizes = result.level_sizes
    if not (sizes[0] > sizes[1] > sizes[2] > sizes[3] > 0):
        raise RuntimeError(f"Invalid four-level graph pyramid sizes: {sizes}")
    return result


def save_surface_graph(graph: SurfaceGraph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "data": graph.data.cpu(),
            "positions": graph.positions.cpu(),
            "original_to_nodes": graph.original_to_nodes.cpu(),
            "original_to_weights": graph.original_to_weights.cpu(),
            "source": graph.source,
        },
        path,
    )


def load_surface_graph(path: str | Path) -> SurfaceGraph:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("version")
    if version not in {1, 2}:
        raise ValueError(f"Unsupported graph cache version in {path}")
    if version == 1:
        original_to_nodes = payload["original_to_node"].to(torch.int32).unsqueeze(1)
        original_to_weights = torch.ones(original_to_nodes.shape, dtype=torch.float32)
    else:
        original_to_nodes = payload["original_to_nodes"]
        original_to_weights = payload["original_to_weights"]
    return SurfaceGraph(
        data=payload["data"],
        positions=payload["positions"],
        original_to_nodes=original_to_nodes,
        original_to_weights=original_to_weights,
        source=payload["source"],
    )
