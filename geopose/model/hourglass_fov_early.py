# stdlib
import os
import sys

# external
import torch
import torch.nn as nn

from geopose.model.hourglass import Module4, Module3, Module2, Module1


class HourglassFovEarly(nn.Module):
    def __init__(self, device):
        super(HourglassFovEarly, self).__init__()
        module_4 = Module4()
        module_3 = Module3(module_4)
        module_2 = Module2(module_3)
        module_1 = Module1(module_2)

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            module_1,
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        self.device = device

    def forward(self, img, fov):
        fov_layer = torch.zeros_like(img[:, 0:1, :, :]) + 1 / fov[:, None, None, None]  # FOV tensor
        fov_layer = fov_layer.to(dtype=torch.half)
        img = torch.cat([img, fov_layer], dim=1)

        out = self.layer(img)
        return out
