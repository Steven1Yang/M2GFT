from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from m2gft.conditioning import FrozenImagePyramid, feature_mean_std, gram_matrix
from m2gft.experiment import DEFAULT_ENCODER

METHODS = ("fss", "m2gft")
DISPLAY_NAMES = {
    "original": "Original",
    "fss": "Standard FSS",
    "m2gft": "M2GFT",
}


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def image_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((lhs - rhs) ** 2))
    return {
        "l1": float(np.mean(np.abs(lhs - rhs))),
        "psnr": float("inf") if mse == 0.0 else -10.0 * math.log10(mse),
        "ssim": float(structural_similarity(lhs, rhs, channel_axis=2, data_range=1.0)),
    }


def appearance_metrics(image: np.ndarray) -> dict[str, float]:
    maximum = image.max(axis=2)
    minimum = image.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 1e-8,
    )
    luminance = (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    )
    dx = np.abs(image[:, 1:, :] - image[:, :-1, :]).mean()
    dy = np.abs(image[1:, :, :] - image[:-1, :, :]).mean()
    return {
        "mean_saturation": float(saturation.mean()),
        "mean_luminance": float(luminance.mean()),
        "luminance_p05": float(np.quantile(luminance, 0.05)),
        "luminance_p95": float(np.quantile(luminance, 0.95)),
        "highlight_fraction": float((luminance >= 0.98).mean()),
        "shadow_fraction": float((luminance <= 0.02).mean()),
        "channel_high_clip_fraction": float((image >= 254.0 / 255.0).mean()),
        "channel_low_clip_fraction": float((image <= 1.0 / 255.0).mean()),
        "mean_abs_gradient": float(0.5 * (dx + dy)),
    }


def as_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(images, axis=0)
    return torch.from_numpy(array).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def compute_vgg_metrics(
    images_by_method: dict[str, list[np.ndarray]],
    methods: tuple[str, ...],
    style_path: Path,
    device: torch.device,
    chunk_size: int,
) -> dict[str, dict[str, float]]:
    encoder = FrozenImagePyramid(DEFAULT_ENCODER).to(device).eval()
    style_np = load_rgb(style_path)
    results = {}
    for method in ("original",) + methods:
        totals = {"gram": 0.0, "stat": 0.0, "content": 0.0}
        count = 0
        for start in range(0, len(images_by_method[method]), chunk_size):
            end = min(start + chunk_size, len(images_by_method[method]))
            output = as_tensor(images_by_method[method][start:end], device)
            content = as_tensor(images_by_method["original"][start:end], device)
            style = as_tensor([style_np], device)
            with torch.no_grad():
                losses = {key: output.new_zeros(()) for key in totals}
                for scale, scale_weight in ((1.0, 1.0), (0.5, 0.5), (0.25, 0.25)):
                    if scale == 1.0:
                        scaled_output, scaled_content, scaled_style = output, content, style
                    else:
                        size = tuple(max(8, round(value * scale)) for value in output.shape[-2:])
                        scaled_output = F.interpolate(output, size=size, mode="bilinear", align_corners=False)
                        scaled_content = F.interpolate(content, size=size, mode="bilinear", align_corners=False)
                        style_size = tuple(max(8, round(value * scale)) for value in style.shape[-2:])
                        scaled_style = F.interpolate(style, size=style_size, mode="bilinear", align_corners=False)
                    output_features = encoder(scaled_output)
                    content_features = encoder(scaled_content)
                    style_features = encoder(scaled_style)
                    for level, layer_weight in (("r11", 1.0), ("r21", 0.8)):
                        output_feature = output_features[level]
                        style_feature = style_features[level]
                        target_gram = gram_matrix(style_feature).mean(dim=0, keepdim=True)
                        losses["gram"] += scale_weight * layer_weight * F.l1_loss(
                            gram_matrix(output_feature),
                            target_gram.expand(output_feature.shape[0], -1, -1),
                        )
                        output_mean, output_std = feature_mean_std(output_feature)
                        style_mean, style_std = feature_mean_std(style_feature)
                        losses["stat"] += scale_weight * layer_weight * (
                            F.l1_loss(output_mean, style_mean.mean(dim=0, keepdim=True).expand_as(output_mean))
                            + F.l1_loss(output_std, style_std.mean(dim=0, keepdim=True).expand_as(output_std))
                        )
                    losses["content"] += scale_weight * F.l1_loss(
                        output_features["r31"], content_features["r31"]
                    )
            batch_size = end - start
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_size
            count += batch_size
            del output, content, style, losses
        results[method] = {key: value / count for key, value in totals.items()}
    return results


def make_contact_sheet(
    renders_dir: Path,
    original_dir: Path,
    style_path: Path,
    out_path: Path,
    num_views: int,
    methods: tuple[str, ...],
    display_names: dict[str, str],
) -> None:
    selected = sorted(set([0, num_views // 4, num_views // 2, 3 * num_views // 4, num_views - 1]))
    cell_width = 500
    gap = 12
    label_height = 42
    columns = ("original",) + methods
    source = Image.open(renders_dir / "view_000" / f"{methods[0]}.png")
    cell_height = round(source.height * cell_width / source.width)
    canvas_width = len(columns) * cell_width + (len(columns) - 1) * gap

    style = Image.open(style_path).convert("RGB")
    style_height = 300
    style_width = round(style.width * style_height / style.height)
    style = style.resize((style_width, style_height), Image.Resampling.LANCZOS)
    header_height = style_height + 95
    row_height = label_height + cell_height + gap
    canvas = Image.new("RGB", (canvas_width, header_height + len(selected) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    except OSError:
        title_font = label_font = ImageFont.load_default()

    draw.text(
        (0, 10),
        f"Style reference: {style_path.stem}",
        fill="black",
        font=title_font,
    )
    canvas.paste(style, (0, 55))
    draw.text(
        (style_width + 35, 160),
        "Same COLMAP test cameras; columns use identical exposure and resolution",
        fill="black",
        font=title_font,
    )

    y = header_height
    for view_idx in selected:
        draw.text((0, y + 5), f"test view {view_idx:03d}", fill="black", font=label_font)
        for column_idx, method in enumerate(columns):
            x = column_idx * (cell_width + gap)
            draw.text((x + 150, y + 5), display_names[method], fill="black", font=label_font)
            if method == "original":
                path = original_dir / f"view_{view_idx:03d}" / "original.png"
            else:
                path = renders_dir / f"view_{view_idx:03d}" / f"{method}.png"
            image = Image.open(path).convert("RGB").resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y + label_height))
        y += row_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders-dir", required=True, type=Path)
    parser.add_argument("--original-dir", required=True, type=Path)
    parser.add_argument("--style", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument(
        "--display-names",
        nargs="+",
        help="Optional display labels corresponding one-to-one with --methods",
    )
    args = parser.parse_args()

    methods = tuple(args.methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("--methods must contain unique method names")
    if args.display_names and len(args.display_names) != len(methods):
        raise ValueError("--display-names must match --methods")
    display_names = {"original": "Original"}
    display_names.update(
        {
            method: label
            for method, label in zip(methods, args.display_names or methods)
        }
    )
    for method, label in DISPLAY_NAMES.items():
        if method in methods and not args.display_names:
            display_names[method] = label

    args.out_dir.mkdir(parents=True, exist_ok=True)
    view_dirs = sorted(args.renders_dir.glob("view_*"))
    if not view_dirs:
        raise FileNotFoundError(f"No view directories under {args.renders_dir}")

    all_methods = ("original",) + methods
    images_by_method = {method: [] for method in all_methods}
    rows = []
    for view_idx, view_dir in enumerate(view_dirs):
        images = {
            "original": load_rgb(args.original_dir / view_dir.name / "original.png"),
            **{method: load_rgb(view_dir / f"{method}.png") for method in methods},
        }
        for method in all_methods:
            images_by_method[method].append(images[method])

        row: dict[str, float | int] = {"view": view_idx}
        for method in methods:
            for metric, value in image_metrics(images[method], images["original"]).items():
                row[f"{method}_vs_original_{metric}"] = value
        for lhs, rhs in itertools.combinations(methods, 2):
            for metric, value in image_metrics(images[lhs], images[rhs]).items():
                row[f"{lhs}_vs_{rhs}_{metric}"] = value
        for method in all_methods:
            for metric, value in appearance_metrics(images[method]).items():
                row[f"{method}_{metric}"] = value
        rows.append(row)

    with (args.out_dir / "metrics_per_view.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        key: summarize([float(row[key]) for row in rows])
        for key in rows[0]
        if key != "view"
    }
    vgg = compute_vgg_metrics(
        images_by_method,
        methods,
        args.style,
        torch.device(args.device),
        args.chunk_size,
    )
    summary = {
        "num_test_views": len(rows),
        "methods": list(methods),
        "style_reference": args.style.name,
        "pixel_and_appearance_metrics": aggregate,
        "vgg_metrics": vgg,
        "interpretation": {
            "pixel_vs_original": "Lower L1 / higher PSNR and SSIM mean more original appearance is retained.",
            "vgg_gram_and_stat": "Style-reference discrepancy measured with the frozen M2GFT evaluation VGG; lower is better.",
            "vgg_content": "VGG r31 discrepancy from the original render; lower is better.",
            "highlight_and_clip_fractions": "Higher values indicate more very bright or channel-clipped pixels.",
            "mean_saturation": "Higher values indicate stronger color saturation.",
            "mean_abs_gradient": "A simple image-space sharpness/edge-strength indicator.",
        },
    }
    (args.out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    make_contact_sheet(
        args.renders_dir,
        args.original_dir,
        args.style,
        args.out_dir / "selected_views_contact_sheet.png",
        len(rows),
        methods,
        display_names,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
