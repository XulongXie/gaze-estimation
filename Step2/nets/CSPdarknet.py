import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import math

# YOLOV4的激活函数 -> Mish()
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, t):
        # Mish 激活函数的公式：x * tanh(ln(1 + e^x))， 其中ln(1 + e^x)又叫softplus
        return t * torch.tanh(F.softplus(t))

# CSPdarknet53的卷积块：Conv2D + BN + Mish
class BasicConv(nn.Module):
    # 卷积块的构造器必须包括一些参数，进行有参构造，否则需要创建很多个卷积块
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super(BasicConv, self).__init__()
        # 卷积
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        # BN
        self.bn = nn.BatchNorm2d(out_channels)
        # Mish
        self.activation = Mish()
    def forward(self, t):
        # 卷积
        t = self.conv(t)
        # BN
        t = self.bn(t)
        # 激活
        t = self.activation(t)
        return t

# CSPdarknet53的残差块：2个卷积层，分别是1*1和3*3的核，并将输出和输入相加
# 卷积层可以直接使用之前的建立的卷积块
class Resblock(nn.Module):
   def __init__(self, channels, hidden_channels = None):
       super(Resblock, self).__init__()
       if (hidden_channels == None):
           hidden_channels = channels
       # 残差块又由一个1*1和一个3*3的卷积块相乘
       self.block = nn.Sequential(BasicConv(channels, hidden_channels, 1), BasicConv(hidden_channels, channels, 3))
   def forward(self, t):
       t = t + self.block(t)
       return t

# CSPdarknet53的Resblock Body：分两条线，一条线由1*1的卷积块和n个残差块最后乘一个1*1的卷积块堆叠而成，另一条线直接用一个1*1的卷积块构成一个很大的残差边
# 相关网络结构图的网址: https://blog.csdn.net/weixin_41560402/article/details/106119774
class Resblock_body(nn.Module):
    # num_bolcks代表着这个resblock body是由几个残差块堆叠而成的，需要注意的是，第一个CSP结构和后面的有略微差别
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        # 首先通过一个卷积进行下采样
        self.down_sampling = BasicConv(in_channels, out_channels, 3, stride = 2)
        # 判断是否是第一个resblock body
        if first:
            # 第一条线，直接做一个大的残差边
            # 首先进行一个1*1的卷积，第一个CSP通道数不减半
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            # 第二条线，残差块会有堆叠，但第一个body只有一个
            self.split_conv2 = BasicConv(out_channels, out_channels, 1)
            # 残差块堆叠后会和一个1*1的卷积块相乘
            self.block = nn.Sequential(Resblock(out_channels, out_channels // 2), BasicConv(out_channels, out_channels, 1))
            # 最后拼接后还会有一个卷积块，第一个CSP拼接后输出通道会变成正常的两倍
            self.last_conv = BasicConv(out_channels * 2, out_channels, 1)
        else:
            # 第一条线，直接做一个大的残差边
            # 首先进行一个1*1的卷积，而且输出通道会减半
            self.split_conv1 = BasicConv(out_channels, out_channels // 2, 1)
            # 第二条线，残差块会有堆叠
            self.split_conv2 = BasicConv(out_channels, out_channels // 2, 1)
            # 残差块堆叠后会和一个1*1的卷积块相乘
            self.block = nn.Sequential(*[Resblock(out_channels // 2) for _ in range (num_blocks)], BasicConv(out_channels // 2, out_channels // 2, 1))
            # 最后拼接后还会有一个卷积块，拼接后输出通道回复正常
            self.last_conv = BasicConv(out_channels, out_channels, 1)
    def forward(self, t):
        # 下采样
        t = self.down_sampling(t)
        # 首先进行第一条线
        t1 = self.split_conv1(t)
        # 接着有第二条线
        t2 = self.split_conv2(t)
        t2 = self.block(t2)
        # 拼接，按照通道数的维度拼接，输出通道数此时还原
        t = torch.cat([t1, t2], dim = 1)
        # 最后一次卷积
        t = self.last_conv(t)
        return t

''' 接下来我们可以构建CSPdarknet的主干了 '''
#   CSPdarknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
class CSPDarkNet(nn.Module):
    # 由于每个resblock body会有不同的堆叠数，所以这里会有一个layers数组
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        # 第一个卷积块416,416,3 -> 416,416,32
        self.conv1 = BasicConv(3, 32, 3)
        # 接下来是残差块，由于残差块有很多而且又是叫同一个名字，所以这里使用nn.ModuleList
        self.stage = nn.ModuleList(
            [# 第一个CSP 416,416,32 -> 208,208,64
            Resblock_body(32, 64, 1, first = True),
            # 第二个CSP 208,208,64 -> 104,104,128
            Resblock_body(64, 128, 2, first = False),
            # 第三个CSP 104,104,128 -> 52,52,256
            Resblock_body(128, 256, 8, first = False),
            # 第四个CSP 52,52,256 -> 26,26,512
            Resblock_body(256, 512, 8, first = False),
            # 第五个CSP 26,26,512 -> 13,13,1024
            Resblock_body(512, 1024, 4, first = False)]
        )

        # 这一块可加可不加，这是在初始化权重
        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 卷积层的权重正态分布
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, t):
        # 卷积块
        t = self.conv1(t)
        # 第1个残差块
        t = self.stage[0](t)
        # 第2个残差块
        t = self.stage[1](t)
        # 第3,4,5个残差块会输出特征层
        out1 = self.stage[2](t)
        out2 = self.stage[3](out1)
        out3 = self.stage[4](out2)
        return out1, out2, out3

# 实例化模型
def darknet53(pretrained, **kwargs):
    # 实例化模型
    model = CSPDarkNet()
    # 如果有预训练过(权重有存储)
    if pretrained:
        # 有预训练的文件就直接加载过来
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model




















