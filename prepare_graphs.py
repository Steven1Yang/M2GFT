#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m2gft.gaussians import load_gaussians
from m2gft.graph import build_surface_graph, save_surface_graph, surface_graph_cache_name


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Precompute the four-resolution surface graphs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scenes.json")
    parser.add_argument("--scenes", nargs="+", default=["garden", "family", "horse", "m60", "train", "truck"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/default/graph_cache")
    parser.add_argument("--max-graph-nodes", type=int, default=60000)
    parser.add_argument("--mapping-neighbors", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2964)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for name in args.scenes:
        cloud = load_gaussians(config[name]["gaussians"])
        print(f"[graph] {name}: sampling from {len(cloud):,} Gaussians")
        graph = build_surface_graph(
            cloud,
            max_nodes=args.max_graph_nodes,
            seed=args.seed,
            build_device=args.device,
            mapping_neighbors=args.mapping_neighbors,
        )
        path = args.output / surface_graph_cache_name(
            name, args.max_graph_nodes, args.seed, args.mapping_neighbors
        )
        save_surface_graph(graph, path)
        print(
            f"[graph] wrote {path}; pyramid={graph.level_sizes}; "
            f"Gaussian interpolation=KNN-{graph.mapping_neighbors}"
        )


if __name__ == "__main__":
    main()
