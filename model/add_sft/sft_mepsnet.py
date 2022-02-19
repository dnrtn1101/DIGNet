import torch
import torch.nn as nn
from torch.nn import init

from model import edsr_ops as ops
from model import share_param as sp


class FeatureEx(nn.Module):
    def __init__(self, feature_size=256):
        super(FeatureEx, self).__init__()
        self.expand_conv1 = nn.Conv2d(3, feature_size//2, 3, 1, 1)  # bias True
        self.expand_conv2 = nn.Conv2d(feature_size//2, feature_size, 3, 1, 1)
        self.conv = nn.Conv2d(feature_size, feature_size, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.expand_conv1(x))
        y = self.relu(self.expand_conv2(y))
        y = self.relu(self.conv(y))
        return y


class Recon(nn.Module):
    def __init__(self, in_channels=64, num_experts=3):
        super(Recon, self).__init__()
        channel_size = in_channels * num_experts
        self.reconst = nn.Sequential(
            nn.Conv2d(channel_size, channel_size//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_size//2, 3, 3, 1, 1)
        )

    def forward(self, x):
        y = self.reconst(x)
        return y


class FeatureFus(nn.Module):
    def __init__(self, feature_size=64, num_experts=3):
        super(FeatureFus, self).__init__()
        channel_size = feature_size * num_experts

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeeze = nn.Conv2d(channel_size, channel_size//2, 1, 1, 0)
        self.excitation = nn.Conv2d(channel_size//2, channel_size, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.relu(self.squeeze(y))
        y = self.sigmoid(self.excitation(y))
        y = x * y
        return y


class Expert(nn.Module):
    def __init__(self,
                 bank, num_templates=16,
                 in_featsize=256, out_featsize=64,
                 num_SRIRs=3, num_SResidualBlocks=12):
        super(Expert, self).__init__()
        self.bank = bank

        head = [
            nn.Conv2d(in_featsize, out_featsize, 3, 1, 1)
        ]

        body = list()
        for _ in range(num_SRIRs):
            body += [sp.SFT_SRIR(self.bank, num_SResidualBlocks)]

        tail = [
            nn.Conv2d(out_featsize, out_featsize, 3, 1, 1)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        # initialize conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)

        # initialize templates in a bank
        total_SRes = num_SRIRs * num_SResidualBlocks  # (3 * 12 => 36)
        total_layers = total_SRes * 2  # ((3 * 12) * 2  => 72)

        coef_inits = torch.zeros(
            (total_layers, num_templates, 1, 1, 1, 1))
        nn.init.orthogonal_(coef_inits)

        # initialize SConv layers
        for i in range(num_SRIRs):
            for j in range(num_SResidualBlocks):
                self.body[i].body[j].conv0.coefficients.data = coef_inits[(
                    i*num_SResidualBlocks)+(2*j)]
                self.body[i].body[j].conv1.coefficients.data = coef_inits[(
                    i*num_SResidualBlocks)+(2*j+1)]

    def forward(self, x, cond):
        x = self.head(x)
        res = self.body[0](x, cond)
        res = self.body[1](res, cond)
        res = self.body[2](res, cond)
        res += x

        y = self.tail(res)
        return y


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        self.sub_mean = ops.MeanShift(rgb_range=255)
        self.add_mean = ops.MeanShift(rgb_range=255, sign=1)

        self.bank = sp.TemplateBank(
            opt.num_templates, opt.expert_featsize, opt.expert_featsize, opt.kernelsize)

        head = [
            FeatureEx(opt.extract_featsize)
        ]

        body = list()
        for _ in range(opt.num_experts):
            body += [Expert(self.bank, opt.num_templates,
                            opt.extract_featsize, opt.expert_featsize, opt.num_SRIRs, opt.num_SResidualBlocks)]

        body += [FeatureFus(opt.expert_featsize, opt.num_experts)]

        tail = [
            Recon(opt.expert_featsize, opt.num_experts)
        ]

        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        self.opt = opt

    def forward(self, x, mask):
        x = self.sub_mean(x)
        cond = self.CondNet(mask)
        x = self.head(x)

        outputs = list()
        for i in range(self.opt.num_experts):
            outputs += [self.body[i](x, cond)]

        x = torch.cat(tuple(outputs), dim=1)
        x = self.body[-1](x)
        x = self.tail(x)
        x = self.add_mean(x)

        return x
