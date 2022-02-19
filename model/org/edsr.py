import torch.nn as nn
from model import edsr_ops as ops


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
            body += [ops.ResBlock(opt.num_channels, opt.res_scale)]
        body += [nn.Conv2d(opt.num_channels, opt.num_channels, 3, 1, 1)]

        tail = [
            nn.Conv2d(opt.num_channels, 3, 3, 1, 1)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        self.opt = opt

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x