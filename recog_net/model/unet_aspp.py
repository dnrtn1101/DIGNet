import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from ops import ops


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        conv = [nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*conv)

        self.down1 = ops.down(64, 128)
        self.down2 = ops.down(128, 256)
        self.down3 = ops.down(256, 512)
        self.down4 = ops.down(512, 1024 // 2, dilation=2)

        self.aspp1 = ops.aspp(512, 128)
        self.aspp2 = ops.aspp(512, 128)
        self.aspp3 = ops.aspp(512, 128)

        self.up1 = ops.up(256, 128 // 2, scale=4)
        self.up2 = ops.up(128, 64, scale=4)
        self.up3 = ops.up(256, 128 // 2, scale=4)
        self.up4 = ops.up(128, 64, scale=4)
        self.up5 = ops.up(256, 128 // 2, scale=4)
        self.up6 = ops.up(128, 64, scale=4)

        self.out1 = nn.Conv2d(64, 10, 3, 1, 1)
        self.out2 = nn.Conv2d(64, 10, 3, 1, 1)
        self.out3 = nn.Conv2d(64, 10, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        blur = self.aspp1(x5)
        blur = self.up1(blur, x2)
        blur = self.up2(blur, x1)
        blur = self.out1(blur)

        noise = self.aspp2(x5)
        noise = self.up3(noise, x2)
        noise = self.up4(noise, x1)
        noise = self.out2(noise)

        jpeg = self.aspp3(x5)
        jpeg = self.up5(jpeg, x2)
        jpeg = self.up6(jpeg, x1)
        jpeg = self.out3(jpeg)

        return blur, noise, jpeg