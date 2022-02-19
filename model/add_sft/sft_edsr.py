import torch.nn as nn
from model import edsr_ops as ops
from model import share_param as sp


class Net(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.sub_mean = ops.MeanShift(255)
        self.add_mean = ops.MeanShift(255, sign=1)

        head = [
            nn.Conv2d(3, opt.num_channels, 3, 1, 1)
        ]

        body = list()
        for _ in range(opt.num_blocks):
            body += [sp.SFT_ResBlock(opt.num_channels)]
        body += [nn.Conv2d(opt.num_channels, opt.num_channels, 3, 1, 1)]

        tail = [
            nn.Conv2d(opt.num_channels, 3, 3, 1, 1)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

        self.opt = opt

    def forward(self, x, mask):
        x = self.sub_mean(x)
        cond = self.CondNet(mask)
        x = self.head(x)
        res = self.body[0](x, cond)
        for i in range(1, len(self.body) - 1):
            res = self.body[i](res, cond)
        res = self.body[-1](res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x