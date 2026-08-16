#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from m2gft.colmap import ColmapScene
from m2gft.checkpoint import ARCHITECTURE_ID
from m2gft.model import M2GFTStylizer
from m2gft.graph import interpolate_node_values
from m2gft.losses import reference_relative_boundary_losses, reference_relative_perceptual_losses
from m2gft.render import render_gaussians
from m2gft.experiment import (
    DEFAULT_DECODER,
    DEFAULT_ENCODER,
    DEFAULT_T41,
    ROOT,
    edge_loss,
    load_style,
    make_scene_state,
    resolve_styles,
    update_latest,
)


ARCHITECTURE = ARCHITECTURE_ID
LOSS_PROTOCOL = "m2gft_spatial_style_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train M2GFT for unseen-scene and unseen-style generalization"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scenes.json")
    parser.add_argument("--scenes", nargs="+", default=["garden", "family", "m60", "train"])
    parser.add_argument("--heldout-scenes", nargs="+", default=["truck", "horse"])
    parser.add_argument("--styles", nargs="+", type=Path)
    parser.add_argument(
        "--style-dir",
        type=Path,
        default=ROOT / "style_ims",
    )
    parser.add_argument("--max-training-styles", type=int, default=60)
    parser.add_argument("--heldout-styles", nargs="+", default=["style1", "style7", "style99"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/m2gft")
    parser.add_argument(
        "--graph-cache-dir",
        type=Path,
        default=ROOT / "runs/graph_cache",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--views-per-step", type=int, default=1)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--style-max-side", type=int, default=512)
    parser.add_argument("--max-graph-nodes", type=int, default=120000)
    parser.add_argument("--mapping-neighbors", type=int, default=4)
    parser.add_argument("--graph-build-device", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--r41", "--t41", dest="r41", type=Path, default=DEFAULT_T41)
    parser.add_argument("--local-blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr-scale", type=float, default=0.05)
    parser.add_argument(
        "--unfreeze-decoder-iteration",
        type=int,
        default=100000,
        help="Optional conv18/conv19 stage; unseen sweep recommends leaving it disabled",
    )

    parser.add_argument("--weight-gram", type=float, default=1.00)
    parser.add_argument("--weight-stats", type=float, default=1.00)
    parser.add_argument("--weight-patch", type=float, default=1.00)
    parser.add_argument("--weight-swd", type=float, default=0.50)
    parser.add_argument("--weight-contextual", type=float, default=1.50)
    parser.add_argument("--weight-color", type=float, default=0.75)
    parser.add_argument("--weight-style-rank", type=float, default=1.00)
    parser.add_argument("--style-rank-margin", type=float, default=0.15)
    parser.add_argument("--style-objective-floor", type=float, default=0.65)

    parser.add_argument("--weight-content", type=float, default=0.50)
    parser.add_argument("--content-guard-tolerance", type=float, default=0.20)
    parser.add_argument("--weight-structure", type=float, default=1.00)
    parser.add_argument("--structure-guard-tolerance", type=float, default=0.15)
    parser.add_argument("--weight-edge", type=float, default=0.10)
    parser.add_argument("--edge-guard-tolerance", type=float, default=0.20)
    parser.add_argument("--weight-range", type=float, default=50.0)
    parser.add_argument("--weight-soft-clip", type=float, default=0.50)
    parser.add_argument("--soft-clip-temperature", type=float, default=0.01)
    parser.add_argument("--weight-distill", type=float, default=0.02)
    parser.add_argument("--distill-iters", type=int, default=10)
    parser.add_argument("--identity-every", type=int, default=32)
    parser.add_argument("--weight-identity", type=float, default=0.02)

    parser.add_argument("--swd-samples", type=int, default=2048)
    parser.add_argument("--swd-projections", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--patch-samples", type=int, default=256)
    parser.add_argument("--patch-style-samples", type=int, default=1024)
    parser.add_argument("--patch-projections", type=int, default=64)
    parser.add_argument("--saliency-fraction", type=float, default=0.70)
    parser.add_argument("--saliency-temperature", type=float, default=1.50)
    parser.add_argument("--contextual-samples", type=int, default=256)
    parser.add_argument("--contextual-style-samples", type=int, default=512)
    parser.add_argument("--color-samples", type=int, default=4096)
    parser.add_argument("--color-projections", type=int, default=16)
    parser.add_argument("--structure-samples", type=int, default=256)

    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2964)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def save_checkpoint(path: Path, model, optimizer, iteration: int, args, latest, styles):
    path.parent.mkdir(parents=True, exist_ok=True)
    decoder_last_trainable = any(
        parameter.requires_grad for parameter in model.decoder.last_backbone_parameters()
    )
    payload = {
        "architecture": ARCHITECTURE,
        "iteration": int(iteration),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "training_scenes": list(args.scenes),
        "heldout_scenes": list(args.heldout_scenes),
        "training_styles": [str(path) for path in styles],
        "heldout_styles": list(args.heldout_styles),
        "latest_losses": latest,
        "frozen_modules": [
            "graph_encoder",
            "style_encoder",
            "r41",
            "decoder_conv11_conv17" if decoder_last_trainable else "decoder_backbone",
        ],
        "decoder_last_trainable": decoder_last_trainable,
        "output_parameterization": "direct_rgb_no_additive_fusion",
        "loss_protocol": LOSS_PROTOCOL,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    for attempt in range(1, 4):
        try:
            torch.save(payload, temporary)
            verified = torch.load(temporary, map_location="cpu", weights_only=False)
            if int(verified.get("iteration", -1)) != int(iteration):
                raise RuntimeError("checkpoint verification returned the wrong iteration")
            os.replace(temporary, path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise


def set_decoder_group_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "decoder_last":
            group["lr"] = float(lr)


def main():
    args = parse_args()
    if not 0.0 < args.style_objective_floor < 1.0:
        raise ValueError("--style-objective-floor must be in (0,1)")
    if set(args.scenes) & set(args.heldout_scenes):
        raise ValueError("Training and held-out scenes overlap")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    styles = resolve_styles(args)
    config = json.loads(args.config.read_text())
    unknown = sorted((set(args.scenes) | set(args.heldout_scenes)) - set(config))
    if unknown:
        raise KeyError(f"Unknown scenes: {unknown}")

    protocol = {
        "architecture": "Multi-level graph pyramid with spatial style-token generation",
        "architecture_id": ARCHITECTURE,
        "output": "direct RGB; no RGB residual and no additive decoder-feature fusion",
        "training_scenes": list(args.scenes),
        "heldout_scenes": list(args.heldout_scenes),
        "training_styles": [path.stem for path in styles],
        "heldout_styles": [Path(name).stem for name in args.heldout_styles],
        "loss_protocol": LOSS_PROTOCOL,
        "style_losses": [
            "all-level Gram",
            "all-level statistics",
            "saliency Patch-SWD",
            "r31 SWD",
            "r11/r21 contextual nearest-neighbour",
            "RGB sliced-OT",
        ],
        "content_losses": ["relative VGG guard", "r31 self-similarity", "edge guard"],
        "pixel_l1_training_weight": 0.0,
        "max_graph_nodes": args.max_graph_nodes,
        "max_image_side": args.max_image_side,
        "training_stages": {
            "pyramid_generator_only_until": args.unfreeze_decoder_iteration - 1,
            "decoder_conv18_conv19_from": args.unfreeze_decoder_iteration,
            "decoder_lr_scale": args.decoder_lr_scale,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))
    if args.dry_run:
        return

    checkpoint = None
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError(f"Not an M2GFT training checkpoint: {args.resume}")
        start_iteration = int(checkpoint["iteration"])

    model = M2GFTStylizer(
        args.encoder,
        args.decoder,
        args.r41,
        local_blocks=args.local_blocks,
    ).to(device).train()
    decoder_enabled = start_iteration >= args.unfreeze_decoder_iteration
    model.set_last_decoder_trainable(decoder_enabled)
    groups = model.optimizer_groups(
        args.lr,
        decoder_lr_scale=args.decoder_lr_scale,
        include_decoder_last=True,
    )
    if not decoder_enabled:
        for group in groups:
            if group["name"] == "decoder_last":
                group["lr"] = 0.0
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=1e-4)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if decoder_enabled:
            set_decoder_group_lr(optimizer, args.lr * args.decoder_lr_scale)
        print(f"[resume] {args.resume}: iteration={start_iteration}")

    style_cache = {path: load_style(path, args.style_max_side, device) for path in styles}
    states = {}
    latest = {}
    style_component_names = (
        "style_gram",
        "style_stats",
        "saliency_patch",
        "swd",
        "contextual",
        "color_ot",
    )
    for iteration in range(start_iteration + 1, args.iterations + 1):
        if iteration == args.unfreeze_decoder_iteration:
            model.set_last_decoder_trainable(True)
            set_decoder_group_lr(optimizer, args.lr * args.decoder_lr_scale)
            print(
                f"[stage] iter={iteration}: enabled decoder conv18/conv19 at "
                f"lr={args.lr * args.decoder_lr_scale:g}"
            )

        scene_name = random.choice(args.scenes)
        if scene_name not in states:
            states[scene_name] = make_scene_state(scene_name, config[scene_name], args, device)
        state = states[scene_name]
        candidates = state.cameras.views("train", max_side=args.max_image_side)
        selected = random.sample(candidates, k=min(args.views_per_step, len(candidates)))
        viewmats, Ks, width, height = ColmapScene.tensors(selected, device)
        style_path = random.choice(styles)
        style = style_cache[style_path]

        details = model(state.graph.data, style, return_details=True)
        output_colors = interpolate_node_values(details["rgb"], state.graph)
        with torch.no_grad():
            reference_colors = interpolate_node_values(details["reference_rgb"], state.graph)
            original_images = render_gaussians(
                state.cloud.means,
                state.cloud.quats,
                state.cloud.scales,
                state.cloud.opacities,
                state.cloud.colors,
                viewmats,
                Ks,
                width,
                height,
            )
            reference_images = render_gaussians(
                state.cloud.means,
                state.cloud.quats,
                state.cloud.scales,
                state.cloud.opacities,
                reference_colors,
                viewmats,
                Ks,
                width,
                height,
            )
        output_images = render_gaussians(
            state.cloud.means,
            state.cloud.quats,
            state.cloud.scales,
            state.cloud.opacities,
            output_colors,
            viewmats,
            Ks,
            width,
            height,
        )
        components = reference_relative_perceptual_losses(
            model.style_encoder,
            output_images,
            reference_images,
            original_images,
            style,
            swd_count=args.swd_samples,
            swd_projections=args.swd_projections,
            patch_size=args.patch_size,
            patch_samples=args.patch_samples,
            patch_style_samples=args.patch_style_samples,
            patch_projections=args.patch_projections,
            saliency_fraction=args.saliency_fraction,
            saliency_temperature=args.saliency_temperature,
            contextual_samples=args.contextual_samples,
            contextual_style_samples=args.contextual_style_samples,
            color_samples=args.color_samples,
            color_projections=args.color_projections,
            structure_samples=args.structure_samples,
        )
        components["pixel"] = F.l1_loss(output_images, original_images)
        components["reference_pixel"] = F.l1_loss(reference_images, original_images).detach()
        components["pixel_ratio"] = components["pixel"] / components["reference_pixel"].clamp_min(1e-6)
        components["edge"] = edge_loss(output_images, original_images)
        components["reference_edge"] = edge_loss(reference_images, original_images).detach()
        components["edge_ratio"] = components["edge"] / components["reference_edge"].clamp_min(1e-6)

        components["content_guard"] = F.relu(
            components["content_ratio"] - (1.0 + args.content_guard_tolerance)
        )
        components["structure_guard"] = F.relu(
            components["structure_ratio"] - (1.0 + args.structure_guard_tolerance)
        )
        components["edge_guard"] = F.relu(
            components["edge_ratio"] - (1.0 + args.edge_guard_tolerance)
        )
        style_ratios = torch.stack(
            [components[f"{name}_ratio"] for name in style_component_names]
        )
        components["style_rank"] = F.relu(
            style_ratios - (1.0 - args.style_rank_margin)
        ).mean()
        for name in style_component_names:
            components[f"{name}_objective"] = F.relu(
                components[f"{name}_ratio"] - args.style_objective_floor
            )
        components.update(
            reference_relative_boundary_losses(
                details["raw_rgb"],
                details["reference_raw"],
                temperature=args.soft_clip_temperature,
            )
        )
        distill_decay = max(
            0.0,
            1.0 - float(iteration - 1) / float(max(args.distill_iters, 1)),
        )
        components["distill"] = F.l1_loss(output_images, reference_images)
        identity_active = args.identity_every > 0 and iteration % args.identity_every == 0
        components["identity"] = output_images.new_zeros(())
        if identity_active:
            identity_rgb = model(state.graph.data, original_images[:1].detach())
            identity_colors = interpolate_node_values(identity_rgb, state.graph)
            identity_image = render_gaussians(
                state.cloud.means,
                state.cloud.quats,
                state.cloud.scales,
                state.cloud.opacities,
                identity_colors,
                viewmats[:1],
                Ks[:1],
                width,
                height,
            )
            components["identity"] = F.l1_loss(identity_image, original_images[:1])

        weights = {
            "style_gram_objective": args.weight_gram,
            "style_stats_objective": args.weight_stats,
            "saliency_patch_objective": args.weight_patch,
            "swd_objective": args.weight_swd,
            "contextual_objective": args.weight_contextual,
            "color_ot_objective": args.weight_color,
            "style_rank": args.weight_style_rank,
            "content_guard": args.weight_content,
            "structure_guard": args.weight_structure,
            "edge_guard": args.weight_edge,
            "range": args.weight_range,
            "soft_clip": args.weight_soft_clip,
            "distill": args.weight_distill * distill_decay,
            "identity": args.weight_identity if identity_active else 0.0,
        }
        total = output_images.new_zeros(())
        for name, weight in weights.items():
            if weight:
                total = total + float(weight) * components[name]

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        latest = {name: float(value.detach()) for name, value in components.items()}
        latest.update(
            {
                "total": float(total.detach()),
                "iteration": iteration,
                "scene": scene_name,
                "style": style_path.stem,
                "views": [camera.name for camera in selected],
                "decoder_last_trainable": any(
                    parameter.requires_grad
                    for parameter in model.decoder.last_backbone_parameters()
                ),
                "raw_out_of_range_fraction": float(
                    ((details["raw_rgb"] < 0.0) | (details["raw_rgb"] > 1.0))
                    .float()
                    .mean()
                    .detach()
                ),
                "m2gft_vs_reference_node_l1": float(
                    F.l1_loss(details["rgb"], details["reference_rgb"]).detach()
                ),
            }
        )
        if iteration == 1 or iteration % args.log_every == 0:
            style_log = " ".join(
                f"{name}={latest[f'{name}_ratio']:.3f}" for name in style_component_names
            )
            print(
                f"[train] iter={iteration} scene={scene_name} style={style_path.stem} "
                f"total={latest['total']:.4f} {style_log} "
                f"content={latest['content_ratio']:.3f} "
                f"structure={latest['structure_ratio']:.3f} edge={latest['edge_ratio']:.3f} "
                f"node_delta={latest['m2gft_vs_reference_node_l1']:.4f} "
                f"range_guard={latest['range']:.5f} "
                f"out_range={latest['raw_out_of_range_fraction']:.4f}"
            )
        if iteration % args.save_every == 0 or iteration == args.iterations:
            checkpoint_path = args.output / f"checkpoint_{iteration:06d}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                iteration,
                args,
                latest,
                styles,
            )
            update_latest(checkpoint_path, args.output / "latest.pt")


if __name__ == "__main__":
    main()
