import torch
import torch.nn as nn
import numpy as np
from model import edsr_ops as ops
from model import share_param as sp
import pdb


class Expert(nn.Module):
    def __init__(self, in_featsize=256, out_featsize=64, num_RIR=2, num_Resblocks=12):
        super(Expert, self).__init__()

        head = [nn.Conv2d(in_featsize, out_featsize, 3, 1, 1)]

        body = []
        for _ in range(num_RIR):
            body += [sp.SFT_RIR(out_featsize, num_Resblocks)]

        tail = [nn.Conv2d(out_featsize, out_featsize, 3, 1, 1)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x, cond):
        x = self.head(x)
        res = self.body[0](x, cond)
        res = self.body[1](res, cond)
        res += x

        y = self.tail(res)

        return y


def transpose_tensor(tensor):
    b, c, w, h = tensor.size()
    tensor = tensor.view(b, c, -1)
    return torch.transpose(tensor, 1, 2)


class Net(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.sub_mean = ops.MeanShift(255)
        self.add_mean = ops.MeanShift(255, sign=1)
        head = [
            nn.Conv2d(3, opt.extract_featsize // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(opt.extract_featsize // 2, opt.extract_featsize, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(opt.extract_featsize, opt.extract_featsize, 3, 1, 1),
            nn.ReLU(inplace=True)
        ]
        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

        self.body = Expert(opt.extract_featsize, opt.expert_featsize, opt.num_SRIRs, opt.num_SResidualBlocks)

        tail = [
            nn.Conv2d(opt.expert_featsize, opt.expert_featsize // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(opt.expert_featsize // 2, 3, 3, 1, 1)
        ]

        self.head = nn.Sequential(*head)
        self.tail = nn.Sequential(*tail)

    def forward(self, x, mask):
        x = self.sub_mean(x)
        cond = self.CondNet(mask)
        res = self.head(x)
        res = self.body(res, cond)
        res = self.tail(res)
        res += x
        res = self.add_mean(res)

        return res