import torch
from torch_scatter import scatter


def max_pool_cluster(x: torch.Tensor, cluster: torch.Tensor) -> torch.Tensor:
    """Max-pool node features according to a precomputed graph cluster."""
    return scatter(x, cluster, dim=0, reduce="max")


def unpool_cluster(x: torch.Tensor, cluster: torch.Tensor) -> torch.Tensor:
    """Broadcast pooled node features back to their source graph level."""
    return torch.index_select(x, 0, cluster)
