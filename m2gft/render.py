from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def render_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    rasterize_mode: str = "antialiased",
) -> torch.Tensor:
    """Differentiably render RGB and composite the result over white."""
    python_bin = str(Path(sys.executable).resolve().parent)
    if python_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
    # M2GFT can point gsplat's JIT loader at the optional project-local toolkit shim
    # prepared by setup_cuda.py.
    local_cuda = Path(__file__).resolve().parents[1] / ".cuda-toolkit"
    if (local_cuda / "bin/nvcc").is_file():
        os.environ.setdefault("CUDA_HOME", str(local_cuda))
        cuda_bins = [str(local_cuda / "nvvm/bin"), str(local_cuda / "bin")]
        current_path = os.environ.get("PATH", "").split(os.pathsep)
        os.environ["PATH"] = os.pathsep.join(cuda_bins + [item for item in current_path if item not in cuda_bins])
        cxx = Path(python_bin) / "x86_64-conda-linux-gnu-c++"
        cc = Path(python_bin) / "x86_64-conda-linux-gnu-cc"
        if cxx.is_file():
            os.environ.setdefault("CXX", str(cxx))
            os.environ.setdefault("NVCC_CCBIN", str(cxx))
        if cc.is_file():
            os.environ.setdefault("CC", str(cc))
    from gsplat.rendering import rasterization

    if viewmats.ndim == 2:
        viewmats = viewmats.unsqueeze(0)
    if Ks.ndim == 2:
        Ks = Ks.unsqueeze(0)
    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=int(width),
        height=int(height),
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        sh_degree=None,
        backgrounds=None,
        render_mode="RGB",
        rasterize_mode=rasterize_mode,
    )
    if render_colors.ndim == 5:
        render_colors = render_colors.squeeze(0)
    if render_alphas is not None and render_alphas.ndim == 5:
        render_alphas = render_alphas.squeeze(0)
    if render_colors.ndim == 3:
        render_colors = render_colors.unsqueeze(0)
    if render_colors.shape[-1] >= 3:
        render_colors = render_colors[..., :3].permute(0, 3, 1, 2)
    elif render_colors.shape[1] != 3:
        raise RuntimeError(f"Unexpected gsplat RGB shape: {tuple(render_colors.shape)}")
    if render_alphas is not None:
        if render_alphas.ndim == 3:
            render_alphas = render_alphas.unsqueeze(0)
        if render_alphas.shape[-1] == 1:
            alpha = render_alphas.permute(0, 3, 1, 2)
        elif render_alphas.shape[1] == 1:
            alpha = render_alphas
        else:
            raise RuntimeError(f"Unexpected gsplat alpha shape: {tuple(render_alphas.shape)}")
        render_colors = render_colors + (1.0 - alpha.clamp(0.0, 1.0))
    return render_colors.clamp(0.0, 1.0)
