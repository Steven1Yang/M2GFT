from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2gft.losses import (
    luminance_collapse_losses,
    reference_relative_boundary_losses,
    reference_relative_perceptual_losses,
)


class TinyPyramid(torch.nn.Module):
    def forward(self, images: torch.Tensor):
        return {
            "r11": images,
            "r21": F.avg_pool2d(images, 2),
            "r31": F.avg_pool2d(images, 4),
            "r41": F.avg_pool2d(images, 8),
        }


def test_equal_output_and_reference_have_unit_ratios_and_gradients():
    torch.manual_seed(7)
    encoder = TinyPyramid()
    reference = torch.rand(1, 3, 32, 32)
    output = reference.clone().requires_grad_(True)
    content = torch.rand_like(reference)
    style = torch.rand(1, 3, 24, 28)
    losses = reference_relative_perceptual_losses(
        encoder,
        output,
        reference,
        content,
        style,
        swd_count=32,
        swd_projections=4,
        patch_samples=8,
        patch_style_samples=16,
        patch_projections=4,
    )
    ratio_names = (
        "style_gram_ratio",
        "style_stats_ratio",
        "saliency_patch_ratio",
        "swd_ratio",
        "contextual_ratio",
        "color_ot_ratio",
        "content_ratio",
        "structure_ratio",
    )
    for name in ratio_names:
        assert torch.allclose(losses[name], torch.ones_like(losses[name]), atol=1e-5), name
    sum(losses[name] for name in ratio_names).backward()
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()
    assert float(output.grad.abs().sum()) > 0.0


def test_boundary_guard_is_zero_at_reference_and_penalizes_only_degradation():
    reference = torch.tensor([[-0.02, 0.25, 1.03]])
    exact = reference.clone().requires_grad_(True)
    losses = reference_relative_boundary_losses(exact, reference)
    assert float(losses["range"]) == 0.0
    assert float(losses["soft_clip"]) == 0.0

    improved = torch.tensor([[0.0, 0.30, 1.0]], requires_grad=True)
    improved_losses = reference_relative_boundary_losses(improved, reference)
    assert float(improved_losses["range"]) == 0.0

    degraded = torch.tensor([[-0.04, 0.30, 1.06]], requires_grad=True)
    degraded_losses = reference_relative_boundary_losses(degraded, reference)
    assert float(degraded_losses["range"]) > 0.0
    assert float(degraded_losses["soft_clip"]) > 0.0
    (degraded_losses["range"] + degraded_losses["soft_clip"]).backward()
    assert degraded.grad is not None


def test_luminance_guard_targets_dark_regions_without_penalizing_safe_output():
    reference = torch.full((1, 3, 32, 32), 0.6)
    content = torch.full_like(reference, 0.5)
    safe = torch.full_like(reference, 0.45, requires_grad=True)
    safe_losses = luminance_collapse_losses(safe, reference, content)
    assert float(safe_losses["shadow_guard"]) == 0.0
    assert float(safe_losses["luminance_guard"]) == 0.0

    collapsed = safe.detach().clone()
    collapsed[:, :, 8:24, 8:24] = 0.02
    collapsed.requires_grad_(True)
    losses = luminance_collapse_losses(collapsed, reference, content)
    assert float(losses["pixel_shadow_guard"]) > 0.0
    assert float(losses["region_shadow_guard"]) > 0.0
    assert float(losses["shadow_deficit_fraction"]) > 0.0
    (losses["shadow_guard"] + losses["luminance_guard"]).backward()
    assert collapsed.grad is not None
    assert float(collapsed.grad[:, :, 8:24, 8:24].abs().sum()) > 0.0


if __name__ == "__main__":
    test_equal_output_and_reference_have_unit_ratios_and_gradients()
    test_boundary_guard_is_zero_at_reference_and_penalizes_only_degradation()
    test_luminance_guard_targets_dark_regions_without_penalizing_safe_output()
    print("reference-relative loss tests passed")
