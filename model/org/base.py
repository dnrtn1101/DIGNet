import torch.nn as nn
from model import edsr_ops as ops
from model import share_param as sp


class Expert(nn.Module):
    def __init__(self, in_featsize=256, out_featsize=64, num_RIR=2, num_Resblocks=12):
        super(Expert, self).__init__()

        head = [nn.Conv2d(in_featsize, out_featsize, 3, 1, 1)]

        body = []
        for _ in range(num_RIR):
            body += [sp.RIR(out_featsize, num_Resblocks)]

        tail = [nn.Conv2d(out_featsize, out_featsize, 3, 1, 1)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x

        y = self.tail(res)

        return y


class Net(nn.Module):
    def __init__(self, opt):
        super().__init__()

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

        body = [Expert(opt.extract_featsize, opt.expert_featsize, opt.num_SRIRs, opt.num_SResidualBlocks)]

        tail = [
            nn.Conv2d(opt.expert_featsize, opt.expert_featsize // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(opt.expert_featsize // 2, 3, 3, 1, 1)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        self.opt = opt

    def forward(self, x):
        x = self.sub_mean(x)

        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        res += x

        res = self.add_mean(res)
        return res