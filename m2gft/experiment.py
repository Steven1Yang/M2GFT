from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .colmap import ColmapScene
from .gaussians import GaussianCloud, load_gaussians
from .graph import (
    SurfaceGraph,
    build_surface_graph,
    load_surface_graph,
    save_surface_graph,
    surface_graph_cache_name,
)


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
MODEL_DIR = WEIGHTS_DIR / "base"
DEFAULT_ENCODER = MODEL_DIR / "vgg_r41.pth"
DEFAULT_DECODER = MODEL_DIR / "dec_r41.pth"
DEFAULT_T41 = MODEL_DIR / "r41.pth"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class SceneState:
    name: str
    cameras: ColmapScene
    cloud: GaussianCloud
    graph: SurfaceGraph


def resolve_model_asset(recorded: str | Path | None, fallback: Path) -> Path:
    path = Path(recorded).expanduser() if recorded else fallback
    return path if path.is_file() else fallback


def same_source_file(left: str | Path, right: str | Path) -> bool:
    """Compare data sources robustly, including hard-linked local dataset aliases."""
    left_path, right_path = Path(left), Path(right)
    try:
        return left_path.samefile(right_path)
    except (FileNotFoundError, OSError):
        return left_path.expanduser().resolve() == right_path.expanduser().resolve()


def resolve_styles(args) -> list[Path]:
    heldout = {Path(name).stem.lower() for name in args.heldout_styles}
    if args.styles:
        candidates = [path.expanduser().resolve() for path in args.styles]
    else:
        candidates = sorted(
            path.resolve()
            for path in args.style_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    candidates = [path for path in candidates if path.stem.lower() not in heldout]
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if not candidates:
        raise ValueError("No training styles remain after applying the held-out split")
    maximum = min(int(args.max_training_styles), len(candidates))
    if maximum < len(candidates):
        indices = np.linspace(0, len(candidates) - 1, maximum, dtype=int)
        candidates = [candidates[index] for index in indices]
    return candidates


def load_style(path: Path, max_side: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_side > 0 and max(image.size) > max_side:
            scale = float(max_side) / float(max(image.size))
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def edge_loss(output: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    dx_output = output[:, :, :, 1:] - output[:, :, :, :-1]
    dx_content = content[:, :, :, 1:] - content[:, :, :, :-1]
    dy_output = output[:, :, 1:, :] - output[:, :, :-1, :]
    dy_content = content[:, :, 1:, :] - content[:, :, :-1, :]
    return F.l1_loss(dx_output, dx_content) + F.l1_loss(dy_output, dy_content)


def make_scene_state(name: str, entry: dict, args, device: torch.device) -> SceneState:
    cameras = ColmapScene(entry["dataset_root"])
    cloud_cpu = load_gaussians(entry["gaussians"], device="cpu")
    cache = args.graph_cache_dir / surface_graph_cache_name(
        name,
        args.max_graph_nodes,
        args.seed,
        args.mapping_neighbors,
    )
    if cache.exists():
        graph = load_surface_graph(cache)
        if not same_source_file(graph.source, cloud_cpu.source):
            raise RuntimeError(f"Graph source mismatch: {cache}")
    else:
        print(f"[graph] building {name} with at most {args.max_graph_nodes:,} nodes")
        graph = build_surface_graph(
            cloud_cpu,
            max_nodes=args.max_graph_nodes,
            seed=args.seed,
            build_device=args.graph_build_device,
            mapping_neighbors=args.mapping_neighbors,
        )
        save_surface_graph(graph, cache)
    print(f"[scene] {name}: pyramid={graph.level_sizes}, Gaussians={len(cloud_cpu):,}")
    return SceneState(name, cameras, cloud_cpu.to(device), graph.to(device))


def update_latest(checkpoint_path: Path, latest_path: Path) -> None:
    """Atomically link latest.pt to an already verified checkpoint."""
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest_path.with_name(f".{latest_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    os.link(checkpoint_path, temporary)
    os.replace(temporary, latest_path)
