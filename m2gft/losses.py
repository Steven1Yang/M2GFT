from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .conditioning import FrozenImagePyramid, feature_mean_std, gram_matrix


STYLE_LEVELS = ("r11", "r21", "r31", "r41")
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _luminance(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected RGB images [B,3,H,W], got {tuple(images.shape)}")
    weights = images.new_tensor(LUMINANCE_WEIGHTS).reshape(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def luminance_collapse_losses(
    output_images: torch.Tensor,
    reference_images: torch.Tensor,
    content_images: torch.Tensor,
    *,
    reference_ratio: float = 0.72,
    content_ratio: float = 0.55,
    maximum_floor: float = 0.30,
    region_kernels: Sequence[int] = (9, 25),
) -> dict[str, torch.Tensor]:
    """Penalize coherent shadow collapse without constraining stylized chroma.

    The guard is one-sided: output that is already bright enough receives no
    penalty. Multi-scale pooling makes a large dark graph region cost more than
    legitimate high-frequency dark strokes from the style image.
    """
    output_luminance = _luminance(output_images)
    with torch.no_grad():
        reference_luminance = _luminance(reference_images)
        content_luminance = _luminance(content_images)
        floor = torch.maximum(
            float(reference_ratio) * reference_luminance,
            float(content_ratio) * content_luminance,
        ).clamp(max=float(maximum_floor))

    pixel_deficit = F.relu(floor - output_luminance)
    region_deficit = output_luminance.new_zeros(())
    valid_kernels = 0
    for raw_kernel in region_kernels:
        kernel = int(raw_kernel)
        if kernel <= 1:
            continue
        if kernel % 2 == 0:
            raise ValueError("Luminance region kernels must be odd")
        pooled_output = F.avg_pool2d(
            output_luminance, kernel, stride=1, padding=kernel // 2
        )
        pooled_floor = F.avg_pool2d(floor, kernel, stride=1, padding=kernel // 2)
        region_deficit = region_deficit + F.relu(pooled_floor - pooled_output).mean()
        valid_kernels += 1
    if valid_kernels:
        region_deficit = region_deficit / float(valid_kernels)

    output_mean = output_luminance.mean(dim=(1, 2, 3))
    reference_mean = reference_luminance.mean(dim=(1, 2, 3))
    content_mean = content_luminance.mean(dim=(1, 2, 3))
    mean_floor = torch.maximum(
        float(reference_ratio) * reference_mean,
        float(content_ratio) * content_mean,
    )
    return {
        "shadow_guard": pixel_deficit.mean() + region_deficit,
        "pixel_shadow_guard": pixel_deficit.mean(),
        "region_shadow_guard": region_deficit,
        "luminance_guard": F.relu(mean_floor - output_mean).mean(),
        "shadow_deficit_fraction": (output_luminance < floor).float().mean().detach(),
        "output_mean_luminance": output_mean.mean().detach(),
        "reference_mean_luminance": reference_mean.mean().detach(),
        "content_mean_luminance": content_mean.mean().detach(),
    }


def reference_relative_boundary_losses(
    raw_rgb: torch.Tensor,
    reference_raw_rgb: torch.Tensor,
    temperature: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Penalize only RGB boundary degradation beyond the frozen graph reference.

    The pretrained reference decoder has a small number of values outside [0, 1].
    An absolute penalty therefore moves the pyramid away from its exact initialization
    before it sees a style gradient. These element-wise guards are zero at the frozen
    reference and react whenever a pyramid value becomes more out-of-range or more
    saturated than the corresponding reference value.
    """
    reference_raw_rgb = reference_raw_rgb.detach()
    output_violation = F.relu(-raw_rgb) + F.relu(raw_rgb - 1.0)
    reference_violation = F.relu(-reference_raw_rgb) + F.relu(reference_raw_rgb - 1.0)
    temperature = max(float(temperature), 1e-4)
    output_boundary = torch.sigmoid(-raw_rgb / temperature) + torch.sigmoid(
        (raw_rgb - 1.0) / temperature
    )
    reference_boundary = torch.sigmoid(-reference_raw_rgb / temperature) + torch.sigmoid(
        (reference_raw_rgb - 1.0) / temperature
    )
    return {
        "range": F.relu(output_violation - reference_violation).mean(),
        "soft_clip": F.relu(output_boundary - reference_boundary).mean(),
        "absolute_range": output_violation.mean(),
        "reference_absolute_range": reference_violation.mean(),
        "absolute_soft_clip": output_boundary.mean(),
        "reference_absolute_soft_clip": reference_boundary.mean(),
    }


def _match_style_gram(output_gram: torch.Tensor, style_gram: torch.Tensor) -> torch.Tensor:
    return style_gram.mean(dim=0, keepdim=True).expand(output_gram.shape[0], -1, -1)


def _resize(images: torch.Tensor, scale: float) -> torch.Tensor:
    if float(scale) == 1.0:
        return images
    height, width = images.shape[-2:]
    return F.interpolate(
        images,
        size=(max(8, round(height * float(scale))), max(8, round(width * float(scale)))),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _saliency_probabilities(feature: torch.Tensor, temperature: float) -> torch.Tensor:
    saliency = feature[:1].detach().std(dim=1, unbiased=False).flatten().clamp_min(0.0)
    saliency = saliency.pow(float(max(temperature, 1e-3)))
    if not torch.isfinite(saliency).all() or float(saliency.sum()) <= 0.0:
        saliency = torch.ones_like(saliency)
    return saliency / saliency.sum().clamp_min(1e-8)


def _mixed_sample_positions(
    feature: torch.Tensor,
    count: int,
    saliency_fraction: float,
    temperature: float,
) -> torch.Tensor:
    if feature.ndim != 4 or feature.shape[0] < 1:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(feature.shape)}")
    _, _, height, width = feature.shape
    count = int(count)
    salient_count = round(count * float(min(max(saliency_fraction, 0.0), 1.0)))
    uniform_count = count - salient_count
    pieces = []
    if salient_count:
        probabilities = _saliency_probabilities(feature, temperature)
        pieces.append(torch.multinomial(probabilities, salient_count, replacement=True))
    if uniform_count:
        pieces.append(torch.randint(height * width, (uniform_count,), device=feature.device))
    indices = torch.cat(pieces, dim=0)
    if len(pieces) > 1:
        indices = indices[torch.randperm(indices.numel(), device=indices.device)]
    ys = (indices // width).float() / float(max(height - 1, 1))
    xs = (indices % width).float() / float(max(width - 1, 1))
    return torch.stack([ys, xs], dim=1).detach()


def _gather_patches(
    feature: torch.Tensor,
    positions: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Gather flattened patches from the first image at normalized positions."""
    if feature.ndim != 4 or feature.shape[0] < 1:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(feature.shape)}")
    _, channels, height, width = feature.shape
    patch_size = int(patch_size)
    padding = patch_size // 2
    padded = F.pad(feature[:1], (padding, padding, padding, padding))
    windows = padded.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    ys = (positions[:, 0] * float(max(height - 1, 1))).round().long().clamp(0, height - 1)
    xs = (positions[:, 1] * float(max(width - 1, 1))).round().long().clamp(0, width - 1)
    patches = windows[0, :, ys, xs].permute(1, 0, 2, 3).contiguous()
    return patches.reshape(patches.shape[0], channels * patch_size * patch_size)


def _resize_sorted_vectors(sorted_vectors: torch.Tensor, count: int) -> torch.Tensor:
    if sorted_vectors.shape[0] == int(count):
        return sorted_vectors
    return F.interpolate(
        sorted_vectors.transpose(0, 1).unsqueeze(0),
        size=int(count),
        mode="linear",
        align_corners=True,
    ).squeeze(0).transpose(0, 1)


def _paired_swd(
    output_vectors: torch.Tensor,
    reference_vectors: torch.Tensor,
    style_vectors: torch.Tensor,
    projections: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute output/style and reference/style SWD with shared random projections."""
    if output_vectors.shape[1] != reference_vectors.shape[1] or output_vectors.shape[1] != style_vectors.shape[1]:
        raise ValueError("SWD feature dimensions must match")
    directions = F.normalize(
        torch.randn(
            output_vectors.shape[1],
            int(projections),
            device=output_vectors.device,
            dtype=output_vectors.dtype,
        ),
        dim=0,
    )
    output_sorted = torch.sort(output_vectors @ directions, dim=0).values
    reference_sorted = torch.sort(reference_vectors @ directions, dim=0).values
    style_sorted = torch.sort(style_vectors @ directions, dim=0).values
    target_count = max(output_sorted.shape[0], style_sorted.shape[0])
    output_sorted = _resize_sorted_vectors(output_sorted, target_count)
    reference_sorted = _resize_sorted_vectors(reference_sorted, target_count)
    style_sorted = _resize_sorted_vectors(style_sorted, target_count)
    return (
        F.l1_loss(output_sorted, style_sorted),
        F.l1_loss(reference_sorted, style_sorted),
    )


def _global_feature_swd(
    output: torch.Tensor,
    reference: torch.Tensor,
    style: torch.Tensor,
    count: int,
    projections: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_vectors = output.permute(0, 2, 3, 1).reshape(-1, output.shape[1])
    reference_vectors = reference.permute(0, 2, 3, 1).reshape(-1, reference.shape[1])
    style_vectors = style.permute(0, 2, 3, 1).reshape(-1, style.shape[1])
    sample_count = min(int(count), output_vectors.shape[0], style_vectors.shape[0])
    output_indices = torch.randperm(output_vectors.shape[0], device=output.device)[:sample_count]
    style_indices = torch.randperm(style_vectors.shape[0], device=style.device)[:sample_count]
    return _paired_swd(
        output_vectors[output_indices],
        reference_vectors[output_indices],
        style_vectors[style_indices],
        projections,
    )


def _paired_contextual_loss(
    output: torch.Tensor,
    reference: torch.Tensor,
    style: torch.Tensor,
    output_count: int,
    style_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour cosine loss using shared output/reference positions."""
    output_vectors = output.permute(0, 2, 3, 1).reshape(-1, output.shape[1])
    reference_vectors = reference.permute(0, 2, 3, 1).reshape(-1, reference.shape[1])
    style_vectors = style.permute(0, 2, 3, 1).reshape(-1, style.shape[1])
    output_count = min(int(output_count), output_vectors.shape[0])
    style_count = min(int(style_count), style_vectors.shape[0])
    output_indices = torch.randperm(output_vectors.shape[0], device=output.device)[:output_count]
    style_indices = torch.randperm(style_vectors.shape[0], device=style.device)[:style_count]
    output_vectors = output_vectors[output_indices]
    reference_vectors = reference_vectors[output_indices]
    style_vectors = style_vectors[style_indices]

    # Center all domains on the style set so cosine similarity represents style patches.
    center = style_vectors.mean(dim=0, keepdim=True)
    output_vectors = F.normalize(output_vectors - center, dim=1, eps=1e-6)
    reference_vectors = F.normalize(reference_vectors - center, dim=1, eps=1e-6)
    style_vectors = F.normalize(style_vectors - center, dim=1, eps=1e-6)
    output_similarity = output_vectors @ style_vectors.transpose(0, 1)
    reference_similarity = reference_vectors @ style_vectors.transpose(0, 1)
    return (
        (1.0 - output_similarity.max(dim=1).values).mean(),
        (1.0 - reference_similarity.max(dim=1).values).mean(),
    )


def _multilevel_contextual_loss(
    output_features: dict[str, torch.Tensor],
    reference_features: dict[str, torch.Tensor],
    style_features: dict[str, torch.Tensor],
    count: int,
    style_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_loss = output_features["r11"].new_zeros(())
    reference_loss = output_features["r11"].new_zeros(())
    weights = {"r11": 1.0, "r21": 0.8}
    weight_sum = sum(weights.values())
    for level, weight in weights.items():
        current, current_reference = _paired_contextual_loss(
            output_features[level],
            reference_features[level],
            style_features[level],
            count,
            style_count,
        )
        output_loss = output_loss + float(weight) * current / weight_sum
        reference_loss = reference_loss + float(weight) * current_reference / weight_sum
    return output_loss, reference_loss


def _color_swd(
    output_images: torch.Tensor,
    reference_images: torch.Tensor,
    style_image: torch.Tensor,
    count: int,
    projections: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = output_images.permute(0, 2, 3, 1).reshape(-1, 3)
    reference = reference_images.permute(0, 2, 3, 1).reshape(-1, 3)
    style = style_image.permute(0, 2, 3, 1).reshape(-1, 3)
    sample_count = min(int(count), output.shape[0], style.shape[0])
    output_indices = torch.randperm(output.shape[0], device=output.device)[:sample_count]
    style_indices = torch.randperm(style.shape[0], device=style.device)[:sample_count]
    return _paired_swd(
        output[output_indices],
        reference[output_indices],
        style[style_indices],
        projections,
    )


def _paired_structure_loss(
    output: torch.Tensor,
    reference: torch.Tensor,
    content: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve spatial feature relationships without forcing content RGB values."""
    output_vectors = output.permute(0, 2, 3, 1).reshape(-1, output.shape[1])
    reference_vectors = reference.permute(0, 2, 3, 1).reshape(-1, reference.shape[1])
    content_vectors = content.permute(0, 2, 3, 1).reshape(-1, content.shape[1])
    sample_count = min(int(count), output_vectors.shape[0])
    indices = torch.randperm(output_vectors.shape[0], device=output.device)[:sample_count]
    output_vectors = F.normalize(output_vectors[indices], dim=1, eps=1e-6)
    reference_vectors = F.normalize(reference_vectors[indices], dim=1, eps=1e-6)
    content_vectors = F.normalize(content_vectors[indices], dim=1, eps=1e-6)
    target = content_vectors @ content_vectors.transpose(0, 1)
    return (
        F.l1_loss(output_vectors @ output_vectors.transpose(0, 1), target),
        F.l1_loss(reference_vectors @ reference_vectors.transpose(0, 1), target),
    )


def _saliency_patch_swd(
    image_encoder: FrozenImagePyramid,
    output_images: torch.Tensor,
    reference_images: torch.Tensor,
    style_image: torch.Tensor,
    full_features: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]],
    layers: Sequence[str],
    scales: Sequence[float],
    scale_weights: Sequence[float],
    layer_weights: dict[str, float],
    patch_size: int,
    output_samples: int,
    style_samples: int,
    projections: int,
    saliency_fraction: float,
    saliency_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_loss = output_images.new_zeros(())
    reference_loss = output_images.new_zeros(())
    scale_weight_sum = float(sum(scale_weights))
    if scale_weight_sum <= 0.0 or len(scales) != len(scale_weights):
        raise ValueError("Patch scales and positive scale weights must match")

    for scale, raw_scale_weight in zip(scales, scale_weights):
        if float(scale) == 1.0:
            output_features, reference_features, style_features = full_features
        else:
            output_features = image_encoder(_resize(output_images, scale))
            with torch.no_grad():
                reference_features = image_encoder(_resize(reference_images, scale))
                style_features = image_encoder(_resize(style_image, scale))

        saliency_layer = layers[0]
        output_positions = _mixed_sample_positions(
            output_features[saliency_layer],
            output_samples,
            saliency_fraction,
            saliency_temperature,
        )
        style_positions = _mixed_sample_positions(
            style_features[saliency_layer],
            style_samples,
            saliency_fraction,
            saliency_temperature,
        )
        layer_output = output_images.new_zeros(())
        layer_reference = output_images.new_zeros(())
        layer_weight_sum = 0.0
        for level in layers:
            output_patches = _gather_patches(output_features[level], output_positions, patch_size)
            reference_patches = _gather_patches(reference_features[level], output_positions, patch_size)
            style_patches = _gather_patches(style_features[level], style_positions, patch_size)
            output_swd, reference_swd = _paired_swd(
                output_patches,
                reference_patches,
                style_patches,
                projections,
            )
            weight = float(layer_weights.get(level, 1.0))
            layer_output = layer_output + weight * output_swd
            layer_reference = layer_reference + weight * reference_swd
            layer_weight_sum += weight
        normalized_scale_weight = float(raw_scale_weight) / scale_weight_sum
        output_loss = output_loss + normalized_scale_weight * layer_output / layer_weight_sum
        reference_loss = reference_loss + normalized_scale_weight * layer_reference / layer_weight_sum
    return output_loss, reference_loss


def reference_relative_perceptual_losses(
    image_encoder: FrozenImagePyramid,
    output_images: torch.Tensor,
    reference_images: torch.Tensor,
    content_images: torch.Tensor,
    style_image: torch.Tensor,
    *,
    swd_count: int = 2048,
    swd_projections: int = 64,
    patch_layers: Sequence[str] = ("r11", "r21"),
    patch_scales: Sequence[float] = (1.0, 0.5),
    patch_scale_weights: Sequence[float] = (1.0, 0.5),
    patch_size: int = 3,
    patch_samples: int = 256,
    patch_style_samples: int = 1024,
    patch_projections: int = 64,
    saliency_fraction: float = 0.7,
    saliency_temperature: float = 1.5,
    contextual_samples: int = 256,
    contextual_style_samples: int = 512,
    color_samples: int = 4096,
    color_projections: int = 16,
    structure_samples: int = 256,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Style losses normalized by a frozen graph reference, plus content guards."""
    output_features = image_encoder(output_images)
    with torch.no_grad():
        reference_features = image_encoder(reference_images)
        content_features = image_encoder(content_images)
        style_features = image_encoder(style_image)

    result: dict[str, torch.Tensor] = {}
    style_gram = output_images.new_zeros(())
    reference_style_gram = output_images.new_zeros(())
    style_stats = output_images.new_zeros(())
    reference_style_stats = output_images.new_zeros(())
    for level in STYLE_LEVELS:
        output = output_features[level]
        reference = reference_features[level]
        style = style_features[level]
        target_gram = gram_matrix(style).mean(dim=0, keepdim=True)
        style_gram = style_gram + F.l1_loss(gram_matrix(output), target_gram.expand(output.shape[0], -1, -1))
        reference_style_gram = reference_style_gram + F.l1_loss(
            gram_matrix(reference), target_gram.expand(reference.shape[0], -1, -1)
        )
        output_mean, output_std = feature_mean_std(output)
        reference_mean, reference_std = feature_mean_std(reference)
        style_mean, style_std = feature_mean_std(style)
        target_mean = style_mean.mean(dim=0, keepdim=True)
        target_std = style_std.mean(dim=0, keepdim=True)
        style_stats = style_stats + F.l1_loss(output_mean, target_mean.expand_as(output_mean))
        style_stats = style_stats + F.l1_loss(output_std, target_std.expand_as(output_std))
        reference_style_stats = reference_style_stats + F.l1_loss(
            reference_mean, target_mean.expand_as(reference_mean)
        )
        reference_style_stats = reference_style_stats + F.l1_loss(
            reference_std, target_std.expand_as(reference_std)
        )

    global_swd, reference_global_swd = _global_feature_swd(
        output_features["r31"],
        reference_features["r31"],
        style_features["r31"],
        swd_count,
        swd_projections,
    )
    saliency_patch, reference_saliency_patch = _saliency_patch_swd(
        image_encoder,
        output_images,
        reference_images,
        style_image,
        (output_features, reference_features, style_features),
        patch_layers,
        patch_scales,
        patch_scale_weights,
        {"r11": 1.0, "r21": 0.8},
        patch_size,
        patch_samples,
        patch_style_samples,
        patch_projections,
        saliency_fraction,
        saliency_temperature,
    )
    contextual, reference_contextual = _multilevel_contextual_loss(
        output_features,
        reference_features,
        style_features,
        contextual_samples,
        contextual_style_samples,
    )
    color_ot, reference_color_ot = _color_swd(
        output_images,
        reference_images,
        style_image,
        color_samples,
        color_projections,
    )

    content = F.l1_loss(output_features["r31"], content_features["r31"])
    content = content + F.l1_loss(output_features["r41"], content_features["r41"])
    reference_content = F.l1_loss(reference_features["r31"], content_features["r31"])
    reference_content = reference_content + F.l1_loss(reference_features["r41"], content_features["r41"])
    structure, reference_structure = _paired_structure_loss(
        output_features["r31"],
        reference_features["r31"],
        content_features["r31"],
        structure_samples,
    )

    pairs = {
        "style_gram": (style_gram, reference_style_gram),
        "style_stats": (style_stats, reference_style_stats),
        "saliency_patch": (saliency_patch, reference_saliency_patch),
        "swd": (global_swd, reference_global_swd),
        "contextual": (contextual, reference_contextual),
        "color_ot": (color_ot, reference_color_ot),
        "content": (content, reference_content),
        "structure": (structure, reference_structure),
    }
    for name, (value, reference_value) in pairs.items():
        result[name] = value
        result[f"reference_{name}"] = reference_value.detach()
        result[f"{name}_ratio"] = value / reference_value.detach().clamp_min(float(epsilon))
    return result
