import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class aspp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(aspp, self).__init__()
        modules = list()
        modules.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1), nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 2, dilation=2), nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 3, dilation=3), nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 4, dilation=4), nn.ReLU(inplace=True)))
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.ReLU(inplace=True))

        self.convs = nn.ModuleList(modules)
        self.proj = nn.Conv2d(out_ch * 5, out_ch, kernel_size=1)

    def forward(self, x):
        size = x.shape[-2:]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False))
        res = torch.cat(res, dim=1)
        return self.proj(res)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(down, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        conv = [nn.Conv2d(in_ch, out_ch, 3, 1, 1, dilation=dilation), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super(up, self).__init__()

        self.upscale = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        conv = [nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // 2, out_ch, 3, 1, 1), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*conv)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)