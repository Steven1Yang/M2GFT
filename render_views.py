#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from m2gft.checkpoint import ARCHITECTURE_ID, load_m2gft_checkpoint
from m2gft.colmap import ColmapScene
from m2gft.experiment import (
    DEFAULT_DECODER,
    DEFAULT_ENCODER,
    DEFAULT_T41,
    load_style,
    resolve_model_asset,
    same_source_file,
)
from m2gft.model import M2GFTStylizer
from m2gft.gaussians import load_gaussians
from m2gft.graph import interpolate_node_values, load_surface_graph, surface_graph_cache_name
from m2gft.render import render_gaussians


ROOT = Path(__file__).resolve().parent


def save_image(path: Path, value: torch.Tensor) -> None:
    array = (
        value.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render M2GFT, with an optional standard-FSS splat comparison"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scenes.json")
    parser.add_argument("--graph-cache", type=Path)
    parser.add_argument(
        "--fss-splat",
        type=Path,
        help="Optional splat produced by the original FastSplatStyler implementation",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE_ID:
        raise ValueError(f"Not an M2GFT checkpoint: {args.checkpoint}")
    train_args = checkpoint["args"]
    config = json.loads(args.config.read_text())
    if args.scene not in config:
        raise KeyError(args.scene)
    entry = config[args.scene]

    max_nodes = int(train_args["max_graph_nodes"])
    seed = int(train_args["seed"])
    mapping_neighbors = int(train_args["mapping_neighbors"])
    if args.graph_cache is None:
        args.graph_cache = Path(train_args["graph_cache_dir"]) / surface_graph_cache_name(
            args.scene, max_nodes, seed, mapping_neighbors
        )
    if not args.graph_cache.is_file():
        raise FileNotFoundError(args.graph_cache)

    device = torch.device(args.device)
    scene = ColmapScene(entry["dataset_root"])
    cameras = scene.views(args.split, max_side=args.max_image_side)
    if args.max_views > 0 and len(cameras) > args.max_views:
        indices = np.linspace(0, len(cameras) - 1, args.max_views, dtype=int)
        cameras = [cameras[index] for index in indices]
    cloud = load_gaussians(entry["gaussians"], device)
    graph = load_surface_graph(args.graph_cache).to(device)
    if not same_source_file(graph.source, cloud.source):
        raise RuntimeError(f"Graph source mismatch: {args.graph_cache}")
    fss_cloud = load_gaussians(args.fss_splat, device) if args.fss_splat else None

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
        m2gft_colors = interpolate_node_values(details["rgb"], graph)

    args.output.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for start in range(0, len(cameras), max(1, args.batch_size)):
        batch = cameras[start : start + max(1, args.batch_size)]
        viewmats, Ks, width, height = ColmapScene.tensors(batch, device)
        with torch.inference_mode():
            original = render_gaussians(
                cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors,
                viewmats, Ks, width, height,
            )
            m2gft = render_gaussians(
                cloud.means, cloud.quats, cloud.scales, cloud.opacities, m2gft_colors,
                viewmats, Ks, width, height,
            )
            fss = None
            if fss_cloud is not None:
                fss = render_gaussians(
                    fss_cloud.means,
                    fss_cloud.quats,
                    fss_cloud.scales,
                    fss_cloud.opacities,
                    fss_cloud.colors,
                    viewmats,
                    Ks,
                    width,
                    height,
                )
        for offset, camera in enumerate(batch):
            view_dir = args.output / f"view_{rendered:03d}"
            save_image(view_dir / "original.png", original[offset])
            if fss is not None:
                save_image(view_dir / "fss.png", fss[offset])
            save_image(view_dir / "m2gft.png", m2gft[offset])
            rendered += 1

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "iteration": int(checkpoint["iteration"]),
        "architecture": checkpoint["architecture"],
        "loss_protocol": checkpoint.get("loss_protocol"),
        "scene": args.scene,
        "scene_was_held_out": args.scene in set(checkpoint.get("heldout_scenes", [])),
        "style": str(args.style.resolve()),
        "fss_splat": str(args.fss_splat.resolve()) if args.fss_splat else None,
        "style_was_held_out": args.style.stem.lower()
        in {Path(value).stem.lower() for value in checkpoint.get("heldout_styles", [])},
        "split": args.split,
        "camera_names": [camera.name for camera in cameras],
        "resolution": [cameras[0].width, cameras[0].height],
        "graph_level_sizes": list(graph.level_sizes),
        "raw_out_of_range_fraction": float(
            ((details["raw_rgb"] < 0.0) | (details["raw_rgb"] > 1.0)).float().mean()
        ),
        "m2gft_vs_reference_node_l1": float(
            (details["rgb"] - details["reference_rgb"]).abs().mean()
        ),
        "num_views": rendered,
        "methods": ["fss", "m2gft"] if args.fss_splat else ["m2gft"],
    }
    (args.output / "render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
