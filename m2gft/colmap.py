from __future__ import annotations

import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


# COLMAP model id -> (name, parameter count).
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_exact(handle, fmt: str):
    size = struct.calcsize("<" + fmt)
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Unexpected end of COLMAP binary while reading {fmt}")
    return struct.unpack("<" + fmt, data)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP's (qw, qx, qy, qz) quaternion to world-to-camera R."""
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec = qvec / max(np.linalg.norm(qvec), 1e-12)
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_cameras_binary(path: str | Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    with Path(path).open("rb") as handle:
        (count,) = _read_exact(handle, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read_exact(handle, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")
            model, parameter_count = CAMERA_MODELS[model_id]
            params = np.asarray(_read_exact(handle, "d" * parameter_count), dtype=np.float64)
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def read_images_binary(path: str | Path) -> list[dict]:
    images = []
    with Path(path).open("rb") as handle:
        (count,) = _read_exact(handle, "Q")
        for _ in range(count):
            values = _read_exact(handle, "idddddddi")
            image_id = int(values[0])
            qvec = np.asarray(values[1:5], dtype=np.float64)
            tvec = np.asarray(values[5:8], dtype=np.float64)
            camera_id = int(values[8])
            name_bytes = bytearray()
            while True:
                char = handle.read(1)
                if not char:
                    raise EOFError("Unexpected end of COLMAP binary while reading image name")
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            (point_count,) = _read_exact(handle, "Q")
            handle.seek(int(point_count) * 24, 1)  # x, y, point3D_id
            images.append(
                {
                    "image_id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name_bytes.decode("utf-8"),
                }
            )
    return images


def _intrinsics(camera: dict) -> np.ndarray:
    model = camera["model"]
    p = camera["params"]
    if model == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy = p[1:3]
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif model in {"SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        fx = fy = p[0]
        cx, cy = p[1:3]
    elif model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError(f"Cannot form pinhole intrinsics for {model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass(frozen=True)
class ColmapCamera:
    image_id: int
    camera_id: int
    name: str
    image_path: Path
    split: str
    width: int
    height: int
    K: torch.Tensor
    viewmat: torch.Tensor
    camera_model: str

    def resized(self, max_side: int | None = None) -> "ColmapCamera":
        if not max_side or max(self.width, self.height) <= max_side:
            return self
        scale = float(max_side) / float(max(self.width, self.height))
        width = max(1, int(round(self.width * scale)))
        height = max(1, int(round(self.height * scale)))
        sx, sy = width / self.width, height / self.height
        K = self.K.clone()
        K[0, :] *= sx
        K[1, :] *= sy
        return replace(self, width=width, height=height, K=K)


class ColmapScene:
    """A scene whose views are read directly from its real COLMAP reconstruction."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        info_path = self.root / "dataset-info.json"
        self.info = json.loads(info_path.read_text()) if info_path.exists() else {}
        loader = self.info.get("loader_kwargs", {})
        self.images_dir = self.root / loader.get("images_path", "images")
        self.model_dir = self.root / loader.get("colmap_path", "sparse/0")
        if not (self.model_dir / "cameras.bin").is_file() or not (self.model_dir / "images.bin").is_file():
            raise FileNotFoundError(f"Missing COLMAP cameras.bin/images.bin under {self.model_dir}")
        self.cameras = self._load()
        if not self.cameras:
            raise RuntimeError(f"No registered cameras found in {self.model_dir}")

    @property
    def name(self) -> str:
        return str(self.info.get("scene", self.root.name))

    def _split_names(self, image_names: list[str]) -> tuple[set[str], set[str]]:
        train_file = self.root / "train_list.txt"
        test_file = self.root / "test_list.txt"
        if train_file.exists() and test_file.exists():
            train = {line.strip() for line in train_file.read_text().splitlines() if line.strip()}
            test = {line.strip() for line in test_file.read_text().splitlines() if line.strip()}
            return train, test
        # Mip-NeRF 360/LLFF convention: every eighth sorted registered view is held out.
        ordered = sorted(image_names)
        test = set(ordered[::8])
        return set(ordered) - test, test

    def _load(self) -> list[ColmapCamera]:
        camera_models = read_cameras_binary(self.model_dir / "cameras.bin")
        images = read_images_binary(self.model_dir / "images.bin")
        train_names, test_names = self._split_names([item["name"] for item in images])
        records = []
        for item in images:
            model = camera_models[item["camera_id"]]
            K = _intrinsics(model)
            viewmat = np.eye(4, dtype=np.float64)
            viewmat[:3, :3] = qvec_to_rotmat(item["qvec"])
            viewmat[:3, 3] = item["tvec"]
            split = "test" if item["name"] in test_names else "train"
            image_path = self.images_dir / item["name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Registered COLMAP image does not exist: {image_path}")
            records.append(
                ColmapCamera(
                    image_id=item["image_id"],
                    camera_id=item["camera_id"],
                    name=item["name"],
                    image_path=image_path,
                    split=split,
                    width=model["width"],
                    height=model["height"],
                    K=torch.from_numpy(K).float(),
                    viewmat=torch.from_numpy(viewmat).float(),
                    camera_model=model["model"],
                )
            )
        return sorted(records, key=lambda record: record.name)

    def views(self, split: str = "train", max_side: int | None = None) -> list[ColmapCamera]:
        if split not in {"train", "test", "all"}:
            raise ValueError(f"split must be train, test, or all; got {split}")
        views = self.cameras if split == "all" else [camera for camera in self.cameras if camera.split == split]
        return [camera.resized(max_side) for camera in views]

    @staticmethod
    def tensors(cameras: Iterable[ColmapCamera], device: str | torch.device = "cpu"):
        cameras = list(cameras)
        if not cameras:
            raise ValueError("At least one camera is required")
        sizes = {(camera.width, camera.height) for camera in cameras}
        if len(sizes) != 1:
            raise ValueError(f"A camera batch must share one resolution, got {sorted(sizes)}")
        viewmats = torch.stack([camera.viewmat for camera in cameras]).to(device)
        Ks = torch.stack([camera.K for camera in cameras]).to(device)
        return viewmats, Ks, cameras[0].width, cameras[0].height

    @staticmethod
    def load_images(cameras: Iterable[ColmapCamera], device: str | torch.device = "cpu") -> torch.Tensor:
        tensors = []
        for camera in cameras:
            with Image.open(camera.image_path) as image:
                image = image.convert("RGB")
                if image.size != (camera.width, camera.height):
                    image = image.resize((camera.width, camera.height), Image.Resampling.LANCZOS)
                array = np.asarray(image, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(tensors).to(device=device, dtype=torch.float32)
