import os
import sys

import torch
import torch.nn as nn


# For NormieNet


#  code taken and slightly modified from: https://github.com/dfan/single-image-surface-normal-estimation
# Implementation of NormieNet: nickname for my altered version of the architecture published by Chen, et al. in NIPS 2016.
# Uses hourglass architecture
class NormieNet(nn.Module):
    def __init__(self):
        super(NormieNet, self).__init__()
        module_4 = Module4()
        module_3 = Module3(module_4)
        module_2 = Module2(module_3)
        module_1 = Module1(module_2)
        self.hourglass = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=1, padding=3),  # toto odpoveda
            nn.BatchNorm2d(128),
            nn.ReLU(),
            module_1,
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)  # tu bude 1 vystupny kanal
        )

    def forward(self, x):
        out = self.hourglass(x)
        return out


class Inception(nn.Module):
    def __init__(self, input_size, output_size, conv_params):
        super(Inception, self).__init__()
        # Base 1 x 1 conv layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU()
        )
        # Additional layer
        self.hidden = nn.ModuleList()
        for i in range(len(conv_params)):
            filt_size = conv_params[i][0]
            pad_size = int((filt_size - 1) / 2)
            out_a = conv_params[i][1]
            out_b = conv_params[i][2]
            curr_layer = nn.Sequential(
                # Reduction
                nn.Conv2d(in_channels=input_size, out_channels=out_a, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_a),
                nn.ReLU(),
                # Spatial convolution
                nn.Conv2d(in_channels=out_a, out_channels=out_b, kernel_size=filt_size, stride=1, padding=pad_size),
                nn.BatchNorm2d(out_b),
                nn.ReLU()
            )
            self.hidden.append(curr_layer)

    def forward(self, x):
        output1 = self.layer1(x)
        outputs = [output1]
        for i in range(len(self.hidden)):
            outputs.append(self.hidden[i](x))
        return torch.cat(outputs, 1)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode=None, align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                      align_corners=self.align_corners)
        return x


class Module1(nn.Module):
    def __init__(self, module_2):
        super(Module1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x
            # tieto 2 su zlte B
            Inception(128, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),
            Inception(128, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),

            module_2,
            # asi zlta B za nou Ruzova A napravo
            Inception(128, 32, [(3, 64, 32), (5, 64, 32), (7, 64, 32)]),
            Inception(128, 16, [(3, 32, 16), (7, 32, 16), (11, 32, 16)]),
            Interpolate(scale_factor=2, mode='nearest')
        )
        self.layer2 = nn.Sequential(
            # input_size, output_size, conv_params [filtsize, out_a 1x1, out_b]
            Inception(128, 16, [(3, 64, 16), (7, 64, 16), (11, 64, 16)])  # ruzove A v nakrese
        )

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(x)
        return output1 + output2


class Module2(nn.Module):
    def __init__(self, module_3):
        super(Module2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x
            # zlte B
            Inception(128, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),
            # Ruzove A
            Inception(128, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),

            module_3,

            # zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            # cervene G
            Inception(256, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),
            Interpolate(scale_factor=2, mode='nearest')  # up to 2x, output is 128 channel
        )
        self.layer2 = nn.Sequential(
            # zlta B
            Inception(128, 32, [(3, 32, 32), (5, 32, 32), (7, 32, 32)]),
            # oranzova C
            Inception(128, 32, [(3, 64, 32), (7, 64, 32), (11, 64, 32)])
        )

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(x)
        return output1 + output2


class Module3(nn.Module):
    def __init__(self, module_4):
        super(Module3, self).__init__()
        self.layer1 = nn.Sequential(
            # zelene E vasie
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            # modre F vacsie
            Inception(256, 64, [(3, 64, 64), (7, 64, 64), (11, 64, 64)])
        )
        self.layer2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),  # 8x
            # zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            # zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),

            module_4,  # down 16x then up to 8x

            # zelene E mensie
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            # modre F mensie
            Inception(256, 64, [(3, 64, 64), (7, 64, 64), (11, 64, 64)]),
            Interpolate(scale_factor=2, mode='nearest')  # up to 4x. 256 channel
        )

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(x)
        return output1 + output2


class Module4(nn.Module):
    def __init__(self):
        super(Module4, self).__init__()
        self.layer1 = nn.Sequential(
            # vsetky zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)])
        )
        self.layer2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            # vsetky zelene E
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            Inception(256, 64, [(3, 32, 64), (5, 32, 64), (7, 32, 64)]),
            Interpolate(scale_factor=2, mode='nearest')  # Up to 8x, 256 channel
        )

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(x)
        return output1 + output2
