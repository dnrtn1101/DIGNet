import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model import owan_ops as ops
from model import share_param as sp
import pdb


class OperationLayer(nn.Module):
    def __init__(self, C, stride):
        super(OperationLayer, self).__init__()

        self._ops = nn.ModuleList()
        for o in ops.Operations:
            op = ops.OPS[o](C, stride, False)
            self._ops.append(op)

        self._out = nn.Sequential(nn.Conv2d(C * len(ops.Operations), C, 1,
                                            padding=0, bias=False), nn.ReLU())

    def forward(self, x, weights):
        weights = weights.transpose(1, 0)
        states = []
        for w, op in zip(weights, self._ops):
            states.append(op(x) * w.view([-1, 1, 1, 1]))

        return self._out(torch.cat(states[:], dim=1))


class GroupOLs(nn.Module):
    def __init__(self, steps, C):
        super(GroupOLs, self).__init__()
        self.preprocess = ops.ReLUConv(C, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        stride = 1
        self.SFT = sp.owan_SFT()
        for _ in range(self._steps):
            op = OperationLayer(C, stride)
            self._ops.append(op)

    def forward(self, s0, cond, weights):
        s0 = self.preprocess(s0)
        for i in range(self._steps):
            res = s0
            s0 = self.SFT(s0, cond)
            s0 = self._ops[i](s0, weights[:, i, :])
            s0 = self.relu(s0 + res)
        return s0


## Operation-wise Attention Layer (OWAL)
class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channel, self.output * 2),
            nn.ReLU(),
            nn.Linear(self.output * 2, self.k * self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y


## entire network (the number of layers = layer_num * steps)
class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.sub_mean = ops.MeanShift(rgb_range=255)
        self.add_mean = ops.MeanShift(rgb_range=255, sign=1)
        self._C = opt.num_ch
        self._layer_num = opt.num_layer
        self._steps = opt.steps
        self.num_ops = len(ops.Operations)

        self.kernel_size = 3
        # Feature Extraction Block
        self.FEB = nn.Sequential(nn.Conv2d(3, self._C, self.kernel_size,
                                           padding=1, bias=False),
                                 sp.owan_SFT_ResBlock(self._C),
                                 sp.owan_SFT_ResBlock(self._C),
                                 sp.owan_SFT_ResBlock(self._C),
                                 sp.owan_SFT_ResBlock(self._C), )

        # a stack of operation-wise attention layers
        self.layers = nn.ModuleList()
        for _ in range(self._layer_num):
            attention = OALayer(self._C, self._steps, self.num_ops)
            self.layers += [attention]
            layer = GroupOLs(self._steps, self._C)
            self.layers += [layer]

        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

        # Output layer
        self.conv2 = nn.Conv2d(self._C, 3, self.kernel_size, padding=1, bias=False)

    def forward(self, x, mask):
        x = self.sub_mean(x)
        cond = self.CondNet(mask)
        s0 = self.FEB[0](x)
        for i in range(1, len(self.FEB)):
            s0 = self.FEB[i](s0, cond)

        for _, layer in enumerate(self.layers):
            if isinstance(layer, OALayer):
                weights = layer(s0)
                weights = F.softmax(weights, dim=-1)
            else:
                s0 = layer(s0, cond, weights)

        logits = self.conv2(s0)
        logits += x
        logits = self.add_mean(logits)

        return logits