from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2gft.colmap import ColmapScene


EXPECTED = {
    "garden": (161, 24),
    "family": (133, 19),
    "horse": (132, 19),
    "m60": (273, 40),
    "train": (263, 38),
    "truck": (219, 32),
}


def test_all_configured_scenes_use_registered_colmap_views():
    local_config = ROOT / "configs/scenes.local.json"
    config_path = local_config if local_config.is_file() else ROOT / "configs/scenes.json"
    config = json.loads(config_path.read_text())
    if any(not Path(config[name]["dataset_root"]).is_dir() for name in EXPECTED):
        print("Skipping real-camera integration test: local datasets are not configured")
        return
    for name, expected_counts in EXPECTED.items():
        scene = ColmapScene(config[name]["dataset_root"])
        train, test = scene.views("train"), scene.views("test")
        assert (len(train), len(test)) == expected_counts
        assert {camera.name for camera in train}.isdisjoint({camera.name for camera in test})
        for camera in (train[0], test[0]):
            assert camera.image_path.is_file()
            assert camera.camera_model == "PINHOLE"
            rotation = camera.viewmat[:3, :3]
            assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-4)
            assert torch.allclose(torch.det(rotation), torch.tensor(1.0), atol=1e-4)
            assert camera.K[0, 0] > 0 and camera.K[1, 1] > 0
        resized = [camera.resized(320) for camera in train[:2]]
        images = scene.load_images(resized)
        assert images.shape == (2, 3, resized[0].height, resized[0].width)
        assert torch.isfinite(images).all() and 0 <= images.min() <= images.max() <= 1


if __name__ == "__main__":
    test_all_configured_scenes_use_registered_colmap_views()
    print("real camera tests passed")
