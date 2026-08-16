#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from m2gft.checkpoint import ARCHITECTURE_ID, load_m2gft_checkpoint
from m2gft.experiment import (
    DEFAULT_DECODER,
    DEFAULT_ENCODER,
    DEFAULT_T41,
    load_style,
    resolve_model_asset,
    same_source_file,
)
from m2gft.model import M2GFTStylizer
from m2gft.gaussians import GaussianCloud, load_gaussians
from m2gft.graph import interpolate_node_values, load_surface_graph, surface_graph_cache_name


ROOT = Path(__file__).resolve().parent


def save_splat(path: Path, cloud: GaussianCloud, colors: torch.Tensor) -> None:
    source = Path(cloud.source)
    if source.suffix.lower() != ".splat":
        raise ValueError("The byte-preserving exporter currently supports .splat sources only")
    dtype = np.dtype(
        [
            ("position", np.float32, 3),
            ("scale", np.float32, 3),
            ("color", np.uint8, 4),
            ("rotation", np.uint8, 4),
        ]
    )
    output = np.fromfile(source, dtype=dtype).copy()
    if len(output) != len(cloud):
        raise ValueError(f"Source splat has {len(output)} records, expected {len(cloud)}")
    rgb = colors.detach().float().cpu().clamp(0.0, 1.0).mul(255.0).round().byte().numpy()
    output["color"][:, :3] = rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    output.tofile(temporary)
    temporary.replace(path)


def stats(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float()
    return {
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export M2GFT colors to a complete .splat")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--scene", default="truck")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scenes.json")
    parser.add_argument("--graph-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE_ID:
        raise ValueError(f"Not an M2GFT checkpoint: {args.checkpoint}")
    train_args = checkpoint["args"]
    config = json.loads(args.config.read_text())
    if args.scene not in config:
        raise KeyError(args.scene)
    scene_config = config[args.scene]
    max_nodes = int(train_args["max_graph_nodes"])
    seed = int(train_args["seed"])
    mapping_neighbors = int(train_args["mapping_neighbors"])
    if args.graph_cache is None:
        cache_root = Path(train_args["graph_cache_dir"])
        args.graph_cache = cache_root / surface_graph_cache_name(
            args.scene, max_nodes, seed, mapping_neighbors
        )
    if not args.graph_cache.is_file():
        raise FileNotFoundError(
            f"Missing held-out graph cache {args.graph_cache}; build it with prepare_graphs.py first"
        )

    device = torch.device(args.device)
    cloud = load_gaussians(scene_config["gaussians"], device)
    graph = load_surface_graph(args.graph_cache).to(device)
    if not same_source_file(graph.source, cloud.source):
        raise RuntimeError(f"Graph source mismatch: {args.graph_cache}")
    model = M2GFTStylizer(
        resolve_model_asset(train_args.get("encoder"), DEFAULT_ENCODER),
        resolve_model_asset(train_args.get("decoder"), DEFAULT_DECODER),
        resolve_model_asset(train_args.get("r41", train_args.get("t41")), DEFAULT_T41),
        local_blocks=int(train_args.get("local_blocks", 2)),
    ).to(device).eval()
    load_m2gft_checkpoint(model, checkpoint)
    style = load_style(args.style, int(train_args.get("style_max_side", 512)), device)
    with torch.inference_mode():
        details = model(graph.data, style, return_details=True)
        colors = interpolate_node_values(details["rgb"], graph)
        reference_colors = interpolate_node_values(details["reference_rgb"], graph)

    save_splat(args.output, cloud, colors)
    heldout_scenes = {str(value) for value in checkpoint.get("heldout_scenes", [])}
    heldout_styles = {Path(value).stem.lower() for value in checkpoint.get("heldout_styles", [])}
    report = {
        "architecture": checkpoint["architecture"],
        "output_parameterization": checkpoint.get("output_parameterization"),
        "loss_protocol": checkpoint.get("loss_protocol", "ema_per_component_v1"),
        "checkpoint": str(args.checkpoint.resolve()),
        "iteration": int(checkpoint["iteration"]),
        "scene": args.scene,
        "scene_was_held_out": args.scene in heldout_scenes,
        "style": str(args.style.resolve()),
        "style_was_held_out": args.style.stem.lower() in heldout_styles,
        "training_scenes": checkpoint.get("training_scenes", []),
        "training_styles": [Path(value).stem for value in checkpoint.get("training_styles", [])],
        "frozen_modules": checkpoint.get("frozen_modules", []),
        "decoder_last_trainable": checkpoint.get("decoder_last_trainable", False),
        "graph_cache": str(args.graph_cache.resolve()),
        "graph_level_sizes": list(graph.level_sizes),
        "mapping_neighbors": graph.mapping_neighbors,
        "gaussians": len(cloud),
        "node_rgb": stats(details["rgb"]),
        "reference_node_rgb": stats(details["reference_rgb"]),
        "m2gft_vs_reference_node_l1": float(
            (details["rgb"] - details["reference_rgb"]).abs().mean()
        ),
        "raw_out_of_range_fraction": float(
            ((details["raw_rgb"] < 0.0) | (details["raw_rgb"] > 1.0)).float().mean()
        ),
        "gaussian_rgb": stats(colors),
        "gaussian_vs_reference_l1": float((colors - reference_colors).abs().mean()),
        "output": str(args.output.resolve()),
    }
    metadata = args.metadata or args.output.with_suffix(".json")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
