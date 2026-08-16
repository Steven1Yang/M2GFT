from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


SH_C0 = 0.28209479177387814


@dataclass
class GaussianCloud:
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    source: str

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def to(self, device: str | torch.device) -> "GaussianCloud":
        return GaussianCloud(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats=self.quats.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
            source=self.source,
        )


def _normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    fallback = np.zeros_like(quats)
    fallback[:, 0] = 1.0
    return np.where(norm > 1e-8, quats / np.maximum(norm, 1e-8), fallback)


def _load_splat(path: Path):
    dtype = np.dtype(
        [
            ("position", np.float32, 3),
            ("scale", np.float32, 3),
            ("color", np.uint8, 4),
            ("rotation", np.uint8, 4),
        ]
    )
    array = np.fromfile(path, dtype=dtype)
    means = np.asarray(array["position"], dtype=np.float32)
    scales = np.asarray(array["scale"], dtype=np.float32)
    colors_rgba = np.asarray(array["color"], dtype=np.float32) / 255.0
    quats = np.asarray(array["rotation"], dtype=np.float32) / 127.5 - 1.0
    return means, scales, _normalize_quaternions(quats), colors_rgba[:, :3], colors_rgba[:, 3]


def _field(vertex, name: str) -> np.ndarray:
    if name not in vertex.data.dtype.names:
        raise KeyError(f"PLY is missing Gaussian field {name!r}")
    return np.asarray(vertex[name])


def _load_ply(path: Path):
    vertex = PlyData.read(path)["vertex"]
    means = np.stack([_field(vertex, key) for key in ("x", "y", "z")], axis=1).astype(np.float32)
    scales_log = np.stack([_field(vertex, key) for key in ("scale_0", "scale_1", "scale_2")], axis=1)
    scales = np.exp(scales_log).astype(np.float32)
    quats = np.stack([_field(vertex, key) for key in ("rot_0", "rot_1", "rot_2", "rot_3")], axis=1)
    quats = _normalize_quaternions(quats.astype(np.float32))
    sh = np.stack([_field(vertex, key) for key in ("f_dc_0", "f_dc_1", "f_dc_2")], axis=1)
    colors = np.clip(0.5 + SH_C0 * sh, 0.0, 1.0).astype(np.float32)
    opacity_logits = _field(vertex, "opacity").astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-opacity_logits))).astype(np.float32)
    return means, scales, quats, colors, opacities


def load_gaussians(path: str | Path, device: str | torch.device = "cpu") -> GaussianCloud:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".splat":
        arrays = _load_splat(path)
    elif path.suffix.lower() == ".ply":
        arrays = _load_ply(path)
    else:
        raise ValueError(f"Expected .splat or .ply, got {path}")
    means, scales, quats, colors, opacities = arrays
    tensors = [torch.from_numpy(np.ascontiguousarray(value)).float() for value in arrays]
    cloud = GaussianCloud(*tensors, source=str(path)).to(device)
    if not all(torch.isfinite(value).all() for value in (cloud.means, cloud.scales, cloud.quats, cloud.colors, cloud.opacities)):
        raise ValueError(f"Non-finite Gaussian values in {path}")
    cloud.scales.clamp_(min=1e-8)
    cloud.colors.clamp_(0.0, 1.0)
    cloud.opacities.clamp_(0.0, 1.0)
    return cloud
