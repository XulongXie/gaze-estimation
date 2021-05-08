from collections import  OrderedDict

import torch
import torch.nn as nn

from Step2.models.Darknet import darknet53

# convolution block, just convert to conv2D + BN + leakyRelu
def conv2d(in_channels, out_channels, kernel_size, stride = 1):
    # padding needs to be set manually
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
                             ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = pad, bias = False)),
                             ("bn", nn.BatchNorm2d(out_channels)),
                             ("relu", nn.LeakyReLU(0.1))
                         ]))

# 3 convolution blocks, 1*1, 3*3, 1*1 kernels, the number of channels: 1024->512->1024->512
def make_three_conv1():
    return nn.Sequential(
                         # 1024->512
                         conv2d(1024, 512, 1),
                         # 512->1024
                         conv2d(512, 1024, 3),
                         # 1024->512
                         conv2d(1024, 512, 1)
    )

# SSP block, the output of the 3 convolution blocks are respectively subjected to 3 × 3, 5 × 5, 7 × 7 maximum pooling,
# and combined with an output without any processing
class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super(SpatialPyramidPooling, self).__init__()
        # class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # 3*3
        self.maxPool1 = nn.MaxPool2d(3, 1, 1)
        # 5*5
        self.maxPool2 = nn.MaxPool2d(5, 1, 2)
        # 7*7
        self.maxPool3 = nn.MaxPool2d(7, 1, 3)

    def forward(self, t):
        out1 = self.maxPool1(t)
        out2 = self.maxPool2(t)
        out3 = self.maxPool3(t)
        out4 = t
        # output channels:　512*4 = 2048
        return torch.cat([out1, out2, out3, out4], dim=1)

# the second cubic convolution block is 1*1, 3*3, 1*1 kernels, and the number of channels is 2048->512->1024->512
def make_three_conv2():
    return nn.Sequential(
                         # channels: 2048->512
                         conv2d(2048, 512, 1),
                         # 512->1024
                         conv2d(512, 1024, 3),
                         # 1024->512
                         conv2d(1024, 512, 1)
    )

# up sampling
class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()
        # the principle of upsampling is to convolve first, then interpolate, and double the width and height
        self.Upsampling = nn.Sequential(
                                        # meaning of convolution is that the number of channels will also change when upsampling
                                        conv2d(in_channels, out_channels, 1),
                                        # upsampling
                                        nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, t):
        return self.Upsampling(t)

# 5 convolution blocks
def make_five_conv(in_channels, out_channels):
    # these five convolutions will halve the number of output channels. The convolution kernels are1*1, 3*3, 1*1, 3*3, 1*1
    return nn.Sequential(
                         conv2d(in_channels, out_channels, kernel_size=1),
                         conv2d(out_channels, in_channels, kernel_size=3),
                         conv2d(in_channels, out_channels, kernel_size=1),
                         conv2d(out_channels, in_channels, kernel_size=3),
                         conv2d(in_channels, out_channels, kernel_size=1)
    )

# feature integration and prediction results
class feature(nn.Module):
    def __init__(self, in_channels, filter_list):
        super(feature, self).__init__()

        self.conv = conv2d(in_channels, filter_list[0], 3)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filter_list[0], filter_list[1])

    def forward(self, t):
        # first is a 3*3 convolution block, 28*28*128 -> 28*28*256
        t = self.conv(t)
        # then a global average pooling, 28*28*256 -> 1*1*256
        t = self.avgPool(t)
        # fully connect layer, output is the 9 classes
        t = t.flatten(start_dim = 1)
        out = self.fc(t)
        return out

# build the network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # first is the backbone we created before for feature extraction
        self.backbone = darknet53(None)

        # 3 conv blocks, 7*7*512
        self.conv1 = make_three_conv1()
        # SSP, 7*7*2048
        self.SSP = SpatialPyramidPooling()
        # another 3 conv blocks, 7*7*512
        self.conv2 = make_three_conv2()

        # 1. upsampling, 7*7*512 -> 14*14*256
        self.upsample1 = upsampling(512, 256)
        # do a convolution on the penultimate feature layer, 14*14*512 -> 14*14*256
        self.conv_second = conv2d(512, 256, 1)
        # 5 convolutions after fusion of the penultimate feature layer, the number of channels after merging is 512, 14*14*512 -> 14*14*256
        self.conv3 = make_five_conv(512, 256)

        # 2. upsampling, 14*14*256 -> 28*28*128
        self.upsample2 = upsampling(256, 128)
        # do a convolution on the penultimate feature layer, 28*28*256 -> 28*28*128
        self.conv_first = conv2d(256, 128, 1)
        # 5 convolutions after the fusion of the first feature layer from the bottom, the number of channels after merging is 256, 28*28*256 -> 28*28*128
        self.conv4 = make_five_conv(256, 128)

        # results under the third scale, 28*28*128 -> 28*28*256 -> 9
        self.output = feature(128, [256, 9])

    def forward(self, t):
        # get the output of the backbone, x0 is the third feature layer everywhere
        x0, x1, x2 = self.backbone(t)

        # for the first feature layer
        out1 = self.conv1(x2)
        out1 = self.SSP(out1)
        out1 = self.conv2(out1)

        # penultimate feature layer
        out2 = self.upsample1(out1)
        P2 = self.conv_second(x1)
        out2 = torch.cat([P2, out2], dim=1)
        out2 = self.conv3(out2)

        # last feature layer
        P1 = self.conv_first(x0)
        out3 = self.upsample2(out2)
        out3 = torch.cat([P1, out3], dim=1)
        out3 = self.conv4(out3)

        # feature output
        predict = out3

        # global average pooling plus full connection
        predict = self.output(predict)

        return predict





















