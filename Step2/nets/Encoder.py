from collections import  OrderedDict

import torch
import torch.nn as nn

from Step2.nets.CSPdarknet import darknet53

# 卷积块，只是换成了conv2D + BN + leakyRelu
def conv2d(in_channels, out_channels, kernel_size, stride = 1):
    # padding需要手动设置
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
                             ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, stride=stride, bias=False)),
                             ("bn", nn.BatchNorm2d(out_channels)),
                             ("leaky", nn.LeakyReLU(0.1))
                         ]))

# 三次卷积块，分别为1*1，3*3,1*1大小的核，通道数1024->512->1024->512
def make_three_conv1():
    return nn.Sequential(
                         # 1024->512
                         conv2d(1024, 512, 1),
                         # 512->1024
                         conv2d(512, 1024, 3),
                         # 1024->512
                         conv2d(1024, 512, 1)
    )

# SSP块，对3次卷积块的输出分别进行5 × 5，9 × 9，13 × 13的最大池化，并且和一个不做任何处理的输出一起合并
class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super(SpatialPyramidPooling, self).__init__()
        # class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # 5*5
        self.maxPool1 = nn.MaxPool2d(5, 1, 2)
        self.maxPool2 = nn.MaxPool2d(9, 1, 4)
        self.maxPool3 = nn.MaxPool2d(13, 1, 6)

    def forward(self, t):
        out1 = self.maxPool1(t)
        out2 = self.maxPool2(t)
        out3 = self.maxPool3(t)
        out4 = t
        # 输出通道数为512*4 = 2048
        return torch.cat([out1, out2, out3, out4], dim=1)

# 第二个三次卷积块，分别为1*1，3*3,1*1大小的核，通道数2048->512->1024->512
def make_three_conv2():
    return nn.Sequential(
                         # 通道数2048->512
                         conv2d(2048, 512, 1),
                         # 512->1024
                         conv2d(512, 1024, 3),
                         # 1024->512
                         conv2d(1024, 512, 1)
    )

# 上采样模块
class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()
        # 上采样是原理是先卷积，再插值，宽高加倍
        self.Upsampling = nn.Sequential(
                                        # 卷积的意义在于上采样时通道数也会变化
                                        conv2d(in_channels, out_channels, 1),
                                        # 上采样
                                        nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, t):
        return self.Upsampling(t)

# 五次卷积模块
def make_five_conv(in_channels, out_channels):
    # 这五次卷积会让输出通道数减半，卷积核分别是1*1, 3*3, 1*1, 3*3, 1*1
    return nn.Sequential(
                         conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1),
                         conv2d(in_channels = out_channels, out_channels = in_channels, kernel_size=3),
                         conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1),
                         conv2d(in_channels = out_channels, out_channels = in_channels, kernel_size=3),
                         conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1)
    )

# 最后的特征整合和预测结果
def feature(in_channels, filter_list):
    return nn.Sequential(
                         # 首先是一个3*3的卷积块
                         conv2d(in_channels, filter_list[0], 3),
                         # 然后是一个单独的卷积层，最后的通道数要符合预测结果需求的通道数
                         nn.Conv2d(filter_list[0], filter_list[1], 1)
    )

# 接下来可以直接创建特征融合的主体结构了
class Feature_Fusion(nn.Module):
    def __init__(self, out_channels):
        super(Feature_Fusion, self).__init__()
        # 首先是我们之前创建的backbone，用于进行特征提取
        self.backbone = darknet53(None)

        # 三次卷积块, 19*19*512
        self.conv1 = make_three_conv1()
        # SSP结构, 19*19*2048
        self.SSP = SpatialPyramidPooling()
        # 第二次经过三次卷积块, 19*19*512
        self.conv2 = make_three_conv2()

        # 第一次上采样, 19*19*512 -> 38*38*256
        self.upsample1 = upsampling(512, 256)
        # 对倒数第二个特征层做一个卷积, 38*38*512 -> 38*38*256
        self.conv_second = conv2d(512, 256, 1)
        # 倒数第二个特征层融合后的五次卷积，合并后通道数为512, 38*38*512 -> 38*38*256
        self.conv3 = make_five_conv(512, 256)

        # 第二次上采样, 38*38*256 -> 76*76*128
        self.upsample2 = upsampling(256, 128)
        # 对倒数第一个特征层做一个卷积, 76*76*256 -> 76*76*128
        self.conv_first = conv2d(256, 128, 1)
        # 倒数第一个特征层融合后的五次卷积，合并后通道数为256, 76*76*256 -> 76*76*128
        self.conv4 = make_five_conv(256, 128)

        # 第一次下采样, 76*76*128 -> 38*38*256
        self.downsample1 = conv2d(128, 256, 3, stride=2)
        # 倒数第二个特征层进行第二次融合后的五次卷积，合并后通道数为512, 38*38*512 -> 38*38*256
        self.conv5 = make_five_conv(512, 256)

        # 第二次下采样, 38*38*256 -> 19*19*512
        self.downsample2 = conv2d(256, 512, 3, stride=2)
        # 倒数第二个特征层进行第三次融合后的五次卷积，合并后通道数为1024, 19*19*1024 -> 19*19*512
        self.conv6 = make_five_conv(1024, 512)

        # 第一个尺度下的预测结果, 19*19*512 -> 19*19*1024 -> 19*19*n
        self.feature1 = feature(512, [1024, out_channels])

        # 第二个尺度下的预测结果, 38*38*256 -> 38*38*512 -> 38*38*n
        self.feature2 = feature(256, [512, out_channels])

        # 第三个尺度下的预测结果, 76*76*128 -> 76*76*256 -> 76*76*n
        self.feature3 = feature(128, [256, out_channels])

    def forward(self, t):
        # 获取backbone的输出, x0是到处第三个特征层
        x0, x1, x2 = self.backbone(t)

        # 首先针对第一个特征层
        out1 = self.conv1(x2)
        out1 = self.SSP(out1)
        out1 = self.conv2(out1)

        # 倒数第二个特征层
        out2 = self.upsample1(out1)
        P2 = self.conv_second(x1)
        out2 = torch.cat([P2, out2], dim=1)
        out2 = self.conv3(out2)

        # 倒数第一个特征层
        P1 = self.conv_first(x0)
        out3 = self.upsample2(out2)
        out3 = torch.cat([P1, out3], dim=1)
        out3 = self.conv4(out3)

        # 倒数第一个输出
        predict3 = out3

        # 倒数第二个输出
        out3 = self.downsample1(out3)
        out3 = torch.cat([out3, out2], dim=1)
        predict2 = self.conv5(out3)

        # 第一个输出
        out2 = self.downsample2(predict2)
        out2 = torch.cat([out2, out1], dim=1)
        predict1 = self.conv6(out2)

        # 整个三个输出
        predict1 = self.feature1(predict1)
        predict2 = self.feature2(predict2)
        predict3 = self.feature3(predict3)

        return predict1, predict2, predict3





















