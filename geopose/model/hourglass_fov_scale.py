# stdlib
import os
import sys

# external
import torch
import torch.nn as nn

from geopose.model.hourglass import Interpolate, Inception
from geopose.model.hourglass_fov import Fovception


class HourglassFovCenterScale(nn.Module):
    def __init__(self):
        super(HourglassFovCenterScale, self).__init__()
        module_4 = FovModule4()
        module_3 = FovModule3(module_4)
        module_2 = FovModule2(module_3)
        module_1 = FovModule1(module_2)

        self.first = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.inner = module_1
        self.last = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, img, fov):
        first_out = self.first(img)
        inner_out, inner_scaling = self.inner(first_out, fov)
        last_out = self.last(inner_out)

        return last_out * inner_scaling[..., None, None], inner_scaling


class FovModule1(nn.Module):
    def __init__(self, module_2):
        super(FovModule1, self).__init__()
        self.first = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x
            # tieto 2 su zlte B
            Inception(128, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),
            Inception(128, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),
        )
        self.inner = module_2

        self.last = nn.Sequential(
            # asi zlta B za nou Ruzova A napravo
            Inception(128, 32, [(3, 64, 32), (5, 64, 32), (7, 64, 32)]),
            Inception(128, 16, [(3, 32, 16), (7, 32, 16), (11, 32, 16)]),
            Interpolate(scale_factor=2, mode='nearest')
        )

        self.residual = nn.Sequential(
            # input_size, output_size, conv_params [filtsize, out_a 1x1, out_b]
            Inception(128, 16, [(3, 64, 16), (7, 64, 16), (11, 64, 16)])  # ruzove A v nakrese
        )

    def forward(self, img, fov):
        first_out = self.first(img)
        inner_out, inner_scaling = self.inner(first_out, fov)
        output1 = self.last(inner_out)

        output2 = self.residual(img)
        # print('1', output1.shape, output2.shape)
        return output1 + output2, inner_scaling


class FovModule2(nn.Module):
    def __init__(self, module_3):
        super(FovModule2, self).__init__()
        self.first = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x
            # zlte B
            Inception(128, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),
            # Ruzove A
            Inception(128, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
        )

        self.last = nn.Sequential(
            # zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            # cervene G
            Inception(256, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),
            Interpolate(scale_factor=2, mode='nearest')  # up to 2x, output is 128 channel
        )

        self.inner = module_3

        self.residual = nn.Sequential(
            # zlta B
            Inception(128, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),
            # oranzova C
            Inception(128, 32, [(3, 64, 32), (7, 64, 32), (11, 64, 32)])
        )

    def forward(self, img, fov):
        first_out = self.first(img)
        inner_out, inner_scaling = self.inner(first_out, fov)
        output1 = self.last(inner_out)

        output2 = self.residual(img)

        return output1 + output2, inner_scaling


class FovModule3(nn.Module):
    def __init__(self, module_4):
        super(FovModule3, self).__init__()

        self.first = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),  # 8x
            # zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            # zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
        )
        self.inner = module_4
        self.last = nn.Sequential(
            # zelene E mensie
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            # modre F mensie
            Inception(256, 64, [(3, 64, 64), (7, 64, 64), (11, 64, 64)]),
            Interpolate(scale_factor=2, mode='nearest')  # up to 4x. 256 channel
        )
        self.residual = nn.Sequential(
            # zelene E vasie
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            # modre F vacsie
            Inception(256, 64, [(3, 64, 64), (7, 64, 64), (11, 64, 64)])
        )

    def forward(self, img, fov):
        first_out = self.first(img)
        inner_out, inner_scaling = self.inner(first_out, fov)
        output1 = self.last(inner_out)

        output2 = self.residual(img)

        return output1 + output2, inner_scaling


class FovModule4(nn.Module):
    def __init__(self):
        super(FovModule4, self).__init__()
        self.residual = nn.Sequential(
            # vsetky zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)])
        )
        self.first = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            # vsetky zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
        )
        self.inner = Fovception(256, 32, [(3, 32, 64), (5, 32, 64), (7, 32, 64)])  # smaller 1x1 conv, FOV concatenated

        self.last = nn.Sequential(
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            Interpolate(scale_factor=2, mode='nearest')  # Up to 8x, 256 channel
        )

        layers = []

        for i in range(0, 4):
            channels_out = 128 // 2 ** i
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            layers.append(nn.Conv2d(in_channels=2 * channels_out, out_channels=channels_out,
                                    kernel_size=1))
            # kernel_size=3, padding=1, padding_mode='reflect'
            layers.append(nn.BatchNorm2d(channels_out))
            layers.append(nn.ReLU())

        self.scale = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(in_features=192, out_features=1)
        )

    def forward(self, img, fov):
        first_out = self.first(img)
        inner_out = self.inner(first_out, fov)
        output1 = self.last(inner_out)

        output2 = self.residual(img)

        out = output1 + output2
        scaling = self.scale(out)

        return out, scaling


def asdf():
    t = torch.zeros((2, 256, 48, 64))

