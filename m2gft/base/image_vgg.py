from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reflect(value: torch.Tensor) -> torch.Tensor:
    return F.pad(value, (1, 1, 1, 1), mode="reflect")


class ImageVGGEncoder(nn.Module):
    """R41 image encoder matching the released pretrained state dictionary."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.conv2 = nn.Conv2d(3, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.conv6 = nn.Conv2d(128, 256, 3)
        self.conv7 = nn.Conv2d(256, 256, 3)
        self.conv8 = nn.Conv2d(256, 256, 3)
        self.conv9 = nn.Conv2d(256, 256, 3)
        self.conv10 = nn.Conv2d(256, 512, 3)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        output = {}
        value = self.conv1(images)
        output["r11"] = F.relu(self.conv2(_reflect(value)), inplace=True)
        output["r12"] = F.relu(self.conv3(_reflect(output["r11"])), inplace=True)
        output["p1"] = F.max_pool2d(output["r12"], 2)
        output["r21"] = F.relu(self.conv4(_reflect(output["p1"])), inplace=True)
        output["r22"] = F.relu(self.conv5(_reflect(output["r21"])), inplace=True)
        output["p2"] = F.max_pool2d(output["r22"], 2)
        output["r31"] = F.relu(self.conv6(_reflect(output["p2"])), inplace=True)
        output["r32"] = F.relu(self.conv7(_reflect(output["r31"])), inplace=True)
        output["r33"] = F.relu(self.conv8(_reflect(output["r32"])), inplace=True)
        output["r34"] = F.relu(self.conv9(_reflect(output["r33"])), inplace=True)
        output["p3"] = F.max_pool2d(output["r34"], 2)
        output["r41"] = F.relu(self.conv10(_reflect(output["p3"])), inplace=True)
        return output


class ImageVGGDecoder(nn.Module):
    """R41 image decoder used to initialize the graph decoder weights."""

    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(512, 256, 3)
        self.conv12 = nn.Conv2d(256, 256, 3)
        self.conv13 = nn.Conv2d(256, 256, 3)
        self.conv14 = nn.Conv2d(256, 256, 3)
        self.conv15 = nn.Conv2d(256, 128, 3)
        self.conv16 = nn.Conv2d(128, 128, 3)
        self.conv17 = nn.Conv2d(128, 64, 3)
        self.conv18 = nn.Conv2d(64, 64, 3)
        self.conv19 = nn.Conv2d(64, 3, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.conv11(_reflect(features)), inplace=True)
        value = F.interpolate(value, scale_factor=2, mode="nearest")
        value = F.relu(self.conv12(_reflect(value)), inplace=True)
        value = F.relu(self.conv13(_reflect(value)), inplace=True)
        value = F.relu(self.conv14(_reflect(value)), inplace=True)
        value = F.relu(self.conv15(_reflect(value)), inplace=True)
        value = F.interpolate(value, scale_factor=2, mode="nearest")
        value = F.relu(self.conv16(_reflect(value)), inplace=True)
        value = F.relu(self.conv17(_reflect(value)), inplace=True)
        value = F.interpolate(value, scale_factor=2, mode="nearest")
        value = F.relu(self.conv18(_reflect(value)), inplace=True)
        return self.conv19(_reflect(value))
