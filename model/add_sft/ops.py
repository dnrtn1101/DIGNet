import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from model import share_param as sp


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range, sign=-1,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0),
    ):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class SFT(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp_gamma1 = nn.Conv2d(32, 64, 1)
        self.mlp_gamma2 = nn.Conv2d(64, 64, 1)
        self.mlp_beta1 = nn.Conv2d(32, 64, 1)
        self.mlp_beta2 = nn.Conv2d(64, 64, 1)

    def forward(self, x, mask):
        gamma = self.mlp_gamma2(F.leaky_relu(self.mlp_gamma1(mask), 0.01, inplace=True))
        beta = self.mlp_beta2(F.leaky_relu(self.mlp_beta1(mask), 0.01, inplace=True))
        out = x * (1 + gamma) + beta

        return out


class SFT_ResBlock(nn.Module):
    def __init__(self, expert_featsize):
        super().__init__()

        self.SFT = SFT()
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(expert_featsize, expert_featsize, 3, 1, 1)
        self.conv1 = nn.Conv2d(expert_featsize, expert_featsize, 3, 1, 1)

    def forward(self, x, cond):
        res = self.SFT(x, cond)
        res = self.relu(self.conv0(res))
        res = self.SFT(res, cond)
        res = self.conv1(res)
        res += x

        return res


class SFT_RIR(nn.Module):
    def __init__(self, expert_featsize, num_Resblocks=12):
        super(SFT_RIR, self).__init__()

        self.num_res = num_Resblocks
        body = list()
        for _ in range(num_Resblocks):
            body += [SFT_ResBlock(expert_featsize)]

        self.body = nn.Sequential(*body)

    def forward(self, x, cond):

        res = self.body[0](x, cond)
        for i in range(1, self.num_res):
            res = self.body[i](res, cond)
        res += x

        return res


class owan_SFT(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp_gamma1 = nn.Conv2d(32, 16, 1)
        self.mlp_gamma2 = nn.Conv2d(16, 16, 1)
        self.mlp_beta1 = nn.Conv2d(32, 16, 1)
        self.mlp_beta2 = nn.Conv2d(16, 16, 1)

    def forward(self, x, mask):
        gamma = self.mlp_gamma2(F.leaky_relu(self.mlp_gamma1(mask), 0.01, inplace=True))
        beta = self.mlp_beta2(F.leaky_relu(self.mlp_beta1(mask), 0.01, inplace=True))
        out = x * (1 + gamma) + beta

        return out


class owan_SFT_ResBlock(nn.Module):
    def __init__(self, expert_featsize):
        super().__init__()

        self.SFT = owan_SFT()
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(expert_featsize, expert_featsize, 3, 1, 1)
        self.conv1 = nn.Conv2d(expert_featsize, expert_featsize, 3, 1, 1)

    def forward(self, x, cond):
        res = self.SFT(x, cond)
        res = self.relu(self.conv0(res))
        res = self.SFT(res, cond)
        res = self.conv1(res)
        res += x

        return res


class TemplateBank(nn.Module):
    def __init__(self, num_templates, in_planes, out_planes, kernel_size):
        super(TemplateBank, self).__init__()

        self.coefficient_shape = (num_templates, 1, 1, 1, 1)
        templates = [torch.Tensor(out_planes, in_planes, kernel_size, kernel_size) for _ in range(num_templates)]
        for i in range(num_templates):
            init.kaiming_normal_(templates[i])
        self.templates = nn.Parameter(torch.stack(templates))

    def forward(self, coefficients):
        return (self.templates * coefficients).sum(0)


class SFT_SRIR(nn.Module):
    def __init__(self, bank, num_SResidualBlocks=12):
        super(SFT_SRIR, self).__init__()

        body = list()
        for _ in range(num_SResidualBlocks):
            body += [SFT_SResidualBlock(bank)]
        self.body = nn.Sequential(*body)

    def forward(self, x, cond):
        res = self.body[0](x, cond)
        for i in range(1, len(self.body)):
            res = self.body[i](res, cond)
        res += x
        return res


class SFT_SResidualBlock(nn.Module):
    def __init__(self, bank):
        super(SFT_SResidualBlock, self).__init__()
        self.SFT = SFT()
        self.conv0 = SConv2d(bank, 1, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = SConv2d(bank, 1, 1, 1)

    def forward(self, x, cond):

        res = self.SFT(x, cond)
        res = self.relu(self.conv0(res))
        res = self.SFT(res, cond)
        res = self.conv1(res)
        res += x

        return res


class SConv2d(nn.Module):
    def __init__(self, bank, stride=1, padding=1, dilation=1):
        super(SConv2d, self).__init__()

        self.stride = stride
        self.padding = padding
        self.bank = bank
        self.dilation = dilation
        self.coefficients = nn.Parameter(torch.zeros(bank.coefficient_shape))

    def forward(self, x):
        params = self.bank(self.coefficients)
        return F.conv2d(x, params, stride=self.stride, padding=self.padding, dilation=self.dilation)


Operations = [
    'sep_conv_1x1',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'dil_conv_7x7',
    'avg_pool_3x3'
]

OPS = {
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'sep_conv_1x1': lambda C, stride, affine: SepConv(C, C, 1, stride, 0, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'dil_conv_7x7': lambda C, stride, affine: DilConv(C, C, 7, stride, 6, 2, affine=affine),
}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                               bias=False)
        self.conv2 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                               bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)

