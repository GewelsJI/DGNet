import random
import numpy as np
import torch
import torch.nn as nn
from lib.EfficientNet import EfficientNet
# from EfficientNet import EfficientNet


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
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


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBNReLU(in_channel, out_channel, 3, padding=1),
            ConvBNReLU(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class MultiGroupAttention(nn.Module):
    def __init__(self, in_channel, out_channel, group_list_N):
        super(MultiGroupAttention, self).__init__()
        self.group_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=group_list_N[0], bias=False)
        self.group_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=group_list_N[1], bias=False)
        self.group_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=group_list_N[2], bias=False)

        pass

        # self.alpha = nn.Parameter(torch.ones(1, out_channel, 1, 1))
        # self.beta = nn.Parameter(torch.ones(1, out_channel, 1, 1))
        # self.gamma = nn.Parameter(torch.ones(1, out_channel, 1, 1))

    def forward(self, x):
        return self.group_conv1(x) + self.group_conv2(x) + self.group_conv3(x)


class GradientInducedTransition(nn.Module):
    def __init__(self, channel, group_list, group_list_N):
        super(GradientInducedTransition, self).__init__()
        self.group_list = group_list

        self.downsample2 = nn.Upsample(scale_factor=1/2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1/4, mode='bilinear', align_corners=True)

        # TODO: some attention modules?
        self.mga3 = MultiGroupAttention(channel+32, channel, group_list_N=group_list_N)
        self.mga4 = MultiGroupAttention(channel+32, channel, group_list_N=group_list_N)
        self.mga5 = MultiGroupAttention(channel+32, channel, group_list_N=group_list_N)
        pass

    def forward(self, x3_rfb, x4_rfb, x5_rfb, grad):
        # transmit the gradient cues into the context embeddings
        x3_rfb_grad = self.split_and_concat(x3_rfb, grad, group=self.group_list[0])
        x4_rfb_grad = self.split_and_concat(x4_rfb, self.downsample2(grad), group=self.group_list[1])
        x5_rfb_grad = self.split_and_concat(x5_rfb, self.downsample4(grad), group=self.group_list[2])

        # attention residual learning
        x3_out = x3_rfb + self.mga3(x3_rfb_grad)
        x4_out = x4_rfb + self.mga4(x4_rfb_grad)
        x5_out = x5_rfb + self.mga5(x5_rfb_grad)

        return x3_out, x4_out, x5_out

    def split_and_concat(self, x, grad, group):
        if group == 1:
            x_cat = torch.cat(
                (x, grad), 1)
        elif group == 2:
            x_g = torch.chunk(x, 2, dim=1)
            grad_g = torch.chunk(grad, 2, dim=1)
            x_cat = torch.cat(
                (x_g[0], grad_g[0], x_g[1], grad_g[1]), 1)
        elif group == 4:
            x_g = torch.chunk(x, 4, dim=1)
            grad_g = torch.chunk(grad, 4, dim=1)
            x_cat = torch.cat(
                (x_g[0], grad_g[0], x_g[1], grad_g[1], x_g[2], grad_g[2], x_g[3], grad_g[3]), 1)
        elif group == 8:
            x_g = torch.chunk(x, 8, dim=1)
            grad_g = torch.chunk(grad, 8, dim=1)
            x_cat = torch.cat(
                (x_g[0], grad_g[0], x_g[1], grad_g[1], x_g[2], grad_g[2], x_g[3], grad_g[3],
                x_g[4], grad_g[4], x_g[5], grad_g[5], x_g[6], grad_g[6], x_g[7], grad_g[7]), 1)
        elif group == 16:
            x_g = torch.chunk(x, 16, dim=1)
            grad_g = torch.chunk(grad, 16, dim=1)
            x_cat = torch.cat(
                (x_g[0], grad_g[0], x_g[1], grad_g[1], x_g[2], grad_g[2], x_g[3], grad_g[3],
                x_g[4], grad_g[4], x_g[5], grad_g[5], x_g[6], grad_g[6], x_g[7], grad_g[7],
                x_g[8], grad_g[8], x_g[9], grad_g[9], x_g[10], grad_g[10], x_g[11], grad_g[11],
                x_g[12], grad_g[12], x_g[13], grad_g[13], x_g[14], grad_g[14], x_g[15], grad_g[15]), 1)
        elif group == 32:
            x_g = torch.chunk(x, 32, dim=1)
            grad_g = torch.chunk(grad, 32, dim=1)
            x_cat = torch.cat(
                (x_g[0], grad_g[0], x_g[1], grad_g[1], x_g[2], grad_g[2], x_g[3], grad_g[3],
                x_g[4], grad_g[4], x_g[5], grad_g[5], x_g[6], grad_g[6], x_g[7], grad_g[7],
                x_g[8], grad_g[8], x_g[9], grad_g[9], x_g[10], grad_g[10], x_g[11], grad_g[11],
                x_g[12], grad_g[12], x_g[13], grad_g[13], x_g[14], grad_g[14], x_g[15], grad_g[15],
                x_g[16], grad_g[16], x_g[17], grad_g[17], x_g[18], grad_g[18], x_g[19], grad_g[19],
                x_g[20], grad_g[20], x_g[21], grad_g[21], x_g[22], grad_g[22], x_g[23], grad_g[23],
                x_g[24], grad_g[24], x_g[25], grad_g[25], x_g[26], grad_g[26], x_g[27], grad_g[27],
                x_g[28], grad_g[28], x_g[29], grad_g[29], x_g[30], grad_g[30], x_g[31], grad_g[31]), 1)
        else:
            raise Exception("Invalid Channel!")

        return x_cat


class NeighborConnectionDecoder(nn.Module):
    """just borrowed from 2021-TPAMI-SINetV2"""
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvBNReLU(channel, channel, 3, padding=1)
        self.conv_upsample2 = ConvBNReLU(channel, channel, 3, padding=1)
        self.conv_upsample3 = ConvBNReLU(channel, channel, 3, padding=1)
        self.conv_upsample4 = ConvBNReLU(channel, channel, 3, padding=1)
        self.conv_upsample5 = ConvBNReLU(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = ConvBNReLU(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = ConvBNReLU(3*channel, 3*channel, 3, padding=1)
        self.conv4 = ConvBNReLU(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat_grad = self.conv3(feat)
        pseudo_grad = self.conv_out(feat_grad)
        return feat_grad, pseudo_grad


class DGNet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, arc='B0', group_list=[8, 8, 8], group_list_N=[4,8,16]):
        super(DGNet, self).__init__()
        self.ch = channel
        self.root = '.'

        if arc == 'B0':
            self.cp = EfficientNet.from_pretrained('efficientnet-b0',
                                                   weights_path=self.root+'/pretrained/efficientnet-b0-355c32eb.pth')
            in_channel_list = [40, 112, 320]
        elif arc == 'B1':
            # self.cp = EfficientNet.from_name('efficientnet-b1')
            self.cp = EfficientNet.from_pretrained('efficientnet-b1',
                                                   weights_path=self.root+'/pretrained/efficientnet-b1-f1951068.pth')
            in_channel_list = [40, 112, 320]
        elif arc == 'B2':
            self.cp = EfficientNet.from_pretrained('efficientnet-b2',
                                                   weights_path=self.root+'/pretrained/efficientnet-b2-8bb594d6.pth')
            in_channel_list = [48, 120, 352]
        elif arc == 'B3':
            self.cp = EfficientNet.from_pretrained('efficientnet-b3',
                                                   weights_path=self.root+'/pretrained/efficientnet-b3-5fb5a3c3.pth')
            in_channel_list = [48, 136, 384]
        elif arc == 'B4':

            # self.cp = EfficientNet.from_name('efficientnet-b4')
            self.cp = EfficientNet.from_pretrained('efficientnet-b4',
                                                   weights_path=self.root+'/pretrained/efficientnet-b4-6ed6700e.pth')
            in_channel_list = [56, 160, 448]
        elif arc == 'B7':
            self.cp = EfficientNet.from_pretrained('efficientnet-b7',
                                                   weights_path=self.root+'/pretrained/efficientnet-b7-dcc49843.pth')
            in_channel_list = [80, 224, 640]
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))

        self.sp = SpatialPath()

        self.red3 = Reduction(in_channel=in_channel_list[0], out_channel=self.ch)
        self.red4 = Reduction(in_channel=in_channel_list[1], out_channel=self.ch)
        self.red5 = Reduction(in_channel=in_channel_list[2], out_channel=self.ch)

        self.git = GradientInducedTransition(channel=self.ch, group_list=group_list, group_list_N=group_list_N)
        self.ncd = NeighborConnectionDecoder(self.ch)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        # context path (encoder)
        endpoints = self.cp.extract_endpoints(x)
        x3 = endpoints['reduction_3']   # torch.Size([1, 40, 44, 44])
        x4 = endpoints['reduction_4']   # torch.Size([1, 112, 22, 22])
        x5 = endpoints['reduction_5']   # torch.Size([1, 320, 11, 11])

        x3_rfb = self.red3(x3)  # torch.Size([1, 32, 44, 44])
        x4_rfb = self.red4(x4)  # torch.Size([1, 32, 22, 22])
        x5_rfb = self.red5(x5)  # torch.Size([1, 32, 11, 11])

        # spatial path (encoder)
        feat_grad, pseudo_grad = self.sp(x)    # torch.Size([1, 1, 44, 44])

        # decoder
        x3_rfb_grad, x4_rfb_grad, x5_rfb_grad = self.git(x3_rfb, x4_rfb, x5_rfb, feat_grad)
        
        final_pred = self.ncd(x5_rfb_grad, x4_rfb_grad, x3_rfb_grad)

        return self.upsample8(final_pred), self.upsample8(pseudo_grad)


if __name__ == '__main__':
    net = DGNet(channel=32, arc='B1', group_list=[8, 8, 8], group_list_N=[4,8,16]).eval()
    inputs = torch.randn(1, 3, 352, 352)
    # net = CubeAttention(45, 8, 45, 9)
    # inputs = torch.randn(1, 45, 44, 44)
    outs = net(inputs)  # torch.Size([1, 45, 44, 44])
    print(outs[0].shape)