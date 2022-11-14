# torch libraries
import torch
import torch.nn as nn
# customized libraries
from .EfficientNet import EfficientNet
from .PVTv2 import *


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class SoftGroupingStrategy(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(SoftGroupingStrategy, self).__init__()
        self.g_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[2], bias=False)

    def forward(self, q):
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)


class GradientInducedTransition(nn.Module):
    def __init__(self, channel, M, N):
        super(GradientInducedTransition, self).__init__()
        self.M = M

        self.downsample2 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=True)

        self.sgs3 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs4 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs5 = SoftGroupingStrategy(channel + 32, channel, N=N)

    def forward(self, xr3, xr4, xr5, xg):
        # transmit the gradient cues into the context embeddings
        q3 = self.gradient_induced_feature_grouping(xr3, xg, M=self.M[0])
        q4 = self.gradient_induced_feature_grouping(xr4, self.downsample2(xg), M=self.M[1])
        q5 = self.gradient_induced_feature_grouping(xr5, self.downsample4(xg), M=self.M[2])

        # attention residual learning
        zt3 = xr3 + self.sgs3(q3)
        zt4 = xr4 + self.sgs4(q4)
        zt5 = xr5 + self.sgs5(q5)

        return zt3, zt4, zt5

    def gradient_induced_feature_grouping(self, xr, xg, M):
        if M == 1:
            q = torch.cat(
                (xr, xg), 1)
        elif M == 2:
            xr_g = torch.chunk(xr, 2, dim=1)
            xg_g = torch.chunk(xg, 2, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1]), 1)
        elif M == 4:
            xr_g = torch.chunk(xr, 4, dim=1)
            xg_g = torch.chunk(xg, 4, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3]), 1)
        elif M == 8:
            xr_g = torch.chunk(xr, 8, dim=1)
            xg_g = torch.chunk(xg, 8, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7]), 1)
        elif M == 16:
            xr_g = torch.chunk(xr, 16, dim=1)
            xg_g = torch.chunk(xg, 16, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7],
                 xr_g[8], xg_g[8], xr_g[9], xg_g[9], xr_g[10], xg_g[10], xr_g[11], xg_g[11],
                 xr_g[12], xg_g[12], xr_g[13], xg_g[13], xr_g[14], xg_g[14], xr_g[15], xg_g[15]), 1)
        elif M == 32:
            xr_g = torch.chunk(xr, 32, dim=1)
            xg_g = torch.chunk(xg, 32, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7],
                 xr_g[8], xg_g[8], xr_g[9], xg_g[9], xr_g[10], xg_g[10], xr_g[11], xg_g[11],
                 xr_g[12], xg_g[12], xr_g[13], xg_g[13], xr_g[14], xg_g[14], xr_g[15], xg_g[15],
                 xr_g[16], xg_g[16], xr_g[17], xg_g[17], xr_g[18], xg_g[18], xr_g[19], xg_g[19],
                 xr_g[20], xg_g[20], xr_g[21], xg_g[21], xr_g[22], xg_g[22], xr_g[23], xg_g[23],
                 xr_g[24], xg_g[24], xr_g[25], xg_g[25], xr_g[26], xg_g[26], xr_g[27], xg_g[27],
                 xr_g[28], xg_g[28], xr_g[29], xg_g[29], xr_g[30], xg_g[30], xr_g[31], xg_g[31]), 1)
        else:
            raise Exception("Invalid Group Number!")

        return q


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample2 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample3 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample4 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample5 = ConvBR(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt5, zt4, zt3):
        zt5_1 = zt5
        zt4_1 = self.conv_upsample1(self.upsample(zt5)) * zt4
        zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) * zt3

        zt4_2 = torch.cat((zt4_1, self.conv_upsample4(self.upsample(zt5_1))), 1)
        zt4_2 = self.conv_concat2(zt4_2)

        zt3_2 = torch.cat((zt3_1, self.conv_upsample5(self.upsample(zt4_2))), 1)
        zt3_2 = self.conv_concat3(zt3_2)

        pc = self.conv4(zt3_2)
        pc = self.conv5(pc)

        return pc


class TextureEncoder(nn.Module):
    def __init__(self):
        super(TextureEncoder, self).__init__()
        self.conv1 = ConvBR(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBR(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBR(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        xg = self.conv3(feat)
        pg = self.conv_out(xg)
        return xg, pg


class DGNet(nn.Module):
    def __init__(self, channel=32, arc='B0', M=[8, 8, 8], N=[4, 8, 16]):
        super(DGNet, self).__init__()
        channel = channel
        # do not load image-net pretrained weights when using ascend model
        if arc == 'EfficientNet-B1':
            print('--> using efficientnet-b1 right now')
            self.context_encoder = EfficientNet.from_pretrained('efficientnet-b1')
            in_channel_list = [40, 112, 320]
        elif arc == 'EfficientNet-B4':
            print('--> using efficientnet-b4 right now')
            self.context_encoder = EfficientNet.from_pretrained('efficientnet-b4')
            in_channel_list = [56, 160, 448]
        elif arc == 'PVTv2-B0':
            print('--> using PVTv2-B0 right now')
            self.context_encoder = pvt_v2_b0(pretrained=False)
            in_channel_list = [64, 160, 256]
        elif arc == 'PVTv2-B1':
            print('--> using PVTv2-B1 right now')
            self.context_encoder = pvt_v2_b1(pretrained=False)
            in_channel_list = [128, 320, 512]
        elif arc == 'PVTv2-B2':
            print('--> using PVTv2-B2 right now')
            self.context_encoder = pvt_v2_b2(pretrained=False)
            in_channel_list = [128, 320, 512]
        elif arc == 'PVTv2-B3':
            print('--> using PVTv2-B3 right now')
            self.context_encoder = pvt_v2_b3(pretrained=False)
            in_channel_list = [128, 320, 512]
        elif arc == 'PVTv2-B4':
            print('--> using PVTv2-B4 right now')
            self.context_encoder = pvt_v2_b4(pretrained=False)
            in_channel_list = [128, 320, 512]
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))

        self.texture_encoder = TextureEncoder()

        self.dr3 = DimensionalReduction(in_channel=in_channel_list[0], out_channel=channel)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[1], out_channel=channel)
        self.dr5 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=channel)

        self.git = GradientInducedTransition(channel=channel, M=M, N=N)
        self.ncd = NeighborConnectionDecoder(channel=channel)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        # context path (encoder)
        endpoints = self.context_encoder.extract_endpoints(x)
        x3 = endpoints['reduction_3']
        x4 = endpoints['reduction_4']
        x5 = endpoints['reduction_5']
        # print(x3.shape, x4.shape, x5.shape)
        xr3 = self.dr3(x3)
        xr4 = self.dr4(x4)
        xr5 = self.dr5(x5)

        # spatial path (encoder)
        xg, pg = self.texture_encoder(x)

        # decoder
        zt3, zt4, zt5 = self.git(xr3, xr4, xr5, xg)

        pc = self.ncd(zt5, zt4, zt3)

        return self.upsample(pc), self.upsample(pg)


if __name__ == '__main__':
    net = DGNet(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).eval()
    inputs = torch.randn(1, 3, 352, 352)
    outs = net(inputs)
    print(outs[0].shape)