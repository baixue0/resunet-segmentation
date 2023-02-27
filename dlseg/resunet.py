import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, initial_bn=True, pool=True, residuals=True):
        super(ResUNetDown, self).__init__()

        if initial_bn:
            self.add_module("norm0", nn.BatchNorm2d(in_channels))

        self.add_module("conv1", nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.add_module("conv2", nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.add_module("shortcut", nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.add_module("norm1", nn.BatchNorm2d(out_channels))
        self.add_module("normShort", nn.BatchNorm2d(out_channels))

        if pool:
            self.add_module("pool", nn.MaxPool2d(2, 2))

        self.activation = torch.relu
        self.residuals = residuals

    def forward(self, image):
        if hasattr(self, "norm0"):
            features = self.norm0(image)
            features = self.activation(image)
        else:
            features = image

        features = self.conv1(features)
        features = self.norm1(features)
        features = self.activation(features)

        features = self.conv2(features)

        if getattr(self, 'residuals', True):
            shortcut = self.shortcut(image)
            shortcut = self.normShort(shortcut)
            skip = features + shortcut
        else:
            skip = features

        if hasattr(self, "pool"):
            pool = self.pool(skip)
            return pool, skip
        else:
            return skip


class ResUNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, residuals=True):
        super(ResUNetUp, self).__init__()

        in_channels = in_channels + out_channels

        self.add_module("upsample", nn.UpsamplingNearest2d(scale_factor=2))
        self.add_module("norm0", nn.BatchNorm2d(in_channels))
        self.add_module("conv1", nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.add_module("conv2", nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.add_module("shortcut", nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.add_module("norm1", nn.BatchNorm2d(out_channels))
        self.add_module("normShort", nn.BatchNorm2d(out_channels))

        self.activation = torch.relu
        self.residuals = residuals

    def forward(self, image, skip):
        image = self.upsample(image)

        image = torch.cat([image, skip], dim=1)

        features = self.norm0(image)
        features = self.activation(features)
        features = self.conv1(features)

        features = self.norm1(features)
        features = self.activation(features)
        features = self.conv2(features)

        if getattr(self, 'residuals', True):
            shortcut = self.shortcut(image)
            shortcut = self.normShort(shortcut)
            features = features + shortcut

        return features


class ResUNet(nn.Module):
    def __init__(self, name, tile_size, in_channels, out_channels, residuals=True):
        super(ResUNet, self).__init__()

        self.down1 = ResUNetDown(in_channels, 12, False, residuals=residuals)
        self.down2 = ResUNetDown(12, 32, residuals=residuals)
        self.down3 = ResUNetDown(32, 64, residuals=residuals)
        self.down4 = ResUNetDown(64, 128, residuals=residuals)

        self.center = ResUNetDown(128, 128, pool=False, residuals=residuals)

        self.up4 = ResUNetUp(128, 128, residuals=residuals)
        self.up3 = ResUNetUp(128, 64, residuals=residuals)
        self.up2 = ResUNetUp(64, 32, residuals=residuals)
        self.up1 = ResUNetUp(32, 12, residuals=residuals)

        self.out = nn.Conv2d(12, out_channels, 1, 1, 0)

        self.name = name
        self.tile_size = tile_size

    def forward(self, image):
        assert image.size(-2) == self.tile_size and image.size(-1) == self.tile_size

        pool1, skip1 = self.down1(image)
        pool2, skip2 = self.down2(pool1)
        pool3, skip3 = self.down3(pool2)
        pool4, skip4 = self.down4(pool3)

        center = self.center(pool4)

        up4 = self.up4(center, skip4)
        up3 = self.up3(up4, skip3)
        up2 = self.up2(up3, skip2)
        up1 = self.up1(up2, skip1)

        return self.out(up1)
