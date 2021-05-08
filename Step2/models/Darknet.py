import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Mish activation function
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, t):
        # Mish function equation：x * tanh(ln(1 + e^x))， with ln(1 + e^x) or softplus
        return t * torch.tanh(F.softplus(t))

# CSPdarknet53 convolution block：Conv2D + BN + Mish
class BasicConv(nn.Module):
    # with some parameters to build this block, otherwise need to build a lot of different blocks
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super(BasicConv, self).__init__()
        # convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        # BN
        self.bn = nn.BatchNorm2d(out_channels)
        # Mish
        self.activation = Mish()
    def forward(self, t):
        # convolution
        t = self.conv(t)
        # BN
        t = self.bn(t)
        # activation
        t = self.activation(t)
        return t

# CSPdarknet53 Resblock：2 convolution layers, with 1*1 and 3*3 kernels, and the input and output together
class Resblock(nn.Module):
   def __init__(self, channels, hidden_channels = None):
       super(Resblock, self).__init__()
       if hidden_channels is None:
           hidden_channels = channels
       # resblock consists of 1*1 and a 3*3 kernel from the convolution block and multiply
       self.block = nn.Sequential(BasicConv(channels, hidden_channels, 1), BasicConv(hidden_channels, channels, 3))
   def forward(self, t):
       t = t + self.block(t)
       return t

# CSPdarknet53 Resblock Body: Divided into two lines, one line is formed by stacking 1 * 1 convolution block and
#                             n residual blocks and finally multiplied by a 1 * 1 convolution block, and the other
#                             line is directly formed by a 1 * 1 convolution block to make a large residual edge
class Resblock_body(nn.Module):
    # num_bolcks represents that this resblock body is made up of several residual blocks stacked.
    # It should be noted that the first CSP structure is slightly different from the following ones
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        # down sampling
        self.down_sampling = BasicConv(in_channels, out_channels, 3, stride = 2)
        # if it's the first resblock body
        if first:
            # first line, to build a large residual edge
            # perform a 1*1 convolution, and the number of channels in the first CSP will not be halved
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            # second line, the residual blocks will be stacked, but the first body has only one
            self.split_conv2 = BasicConv(out_channels, out_channels, 1)
            # after the residual block is stacked, it will be multiplied with a 1*1 convolution block
            self.block = nn.Sequential(Resblock(out_channels, out_channels // 2), BasicConv(out_channels, out_channels, 1))
            # after the final splicing, there will be a convolution block. After the concatation, the output channel will become twice as normal
            self.last_conv = BasicConv(out_channels * 2, out_channels, 1)
        else:
            # first line, to build a large residual edge
            # perform a 1*1 convolution, and the number of channels will be halved
            self.split_conv1 = BasicConv(out_channels, out_channels // 2, 1)
            # second line, the residual blocks will be stacked
            self.split_conv2 = BasicConv(out_channels, out_channels // 2, 1)
            # after the residual block is stacked, it will be multiplied with a 1*1 convolution block
            self.block = nn.Sequential(*[Resblock(out_channels // 2) for _ in range (num_blocks)], BasicConv(out_channels // 2, out_channels // 2, 1))
            # after the final splicing, there will be a convolution block. After the concatenation, the output channel will become normal
            self.last_conv = BasicConv(out_channels, out_channels, 1)
    def forward(self, t):
        # down sampling
        t = self.down_sampling(t)
        # first line
        t1 = self.split_conv1(t)
        # second line
        t2 = self.split_conv2(t)
        t2 = self.block(t2)
        # concatenation
        t = torch.cat([t1, t2], dim = 1)
        # last convolution
        t = self.last_conv(t)
        return t

''' Next we can build the backbone of CSPdarknet'''
# The main part of CSPdarknet53
# Input as a 224x224x1 picture
# Output is three effective feature layers
class CSPDarkNet(nn.Module):
    # since each resblock body will have a different stacking number, there will be a layers array here
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        # first convolution block 224,224,1 -> 224,224,32
        self.conv1 = BasicConv(1, 32, kernel_size = 3, stride = 1)
        # next is the residual block, because there are many residual blocks and they have the same name, so nn.ModuleList is used here
        self.stage = nn.ModuleList(
            [# first CSP 224,224,32 -> 112,112,64
             Resblock_body(32, 64, 1, first = True),
             # second CSP 112,112,64 -> 56,56,128
             Resblock_body(64, 128, 2, first = False),
             # third CSP 56,56,128 -> 28,28,256
             Resblock_body(128, 256, 8, first = False),
             # forth CSP 28,28,256 -> 14,14,512
             Resblock_body(256, 512, 8, first = False),
             # fifth CSP 14,14,512 -> 7,7,1024
             Resblock_body(512, 1024, 4, first = False)]
        )

        # this piece can be added or not, this is to initialize the weight
        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # the weights of the convolutional layer are normally distributed
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, t):
        # convolution block
        t = self.conv1(t)
        # first resblock
        t = self.stage[0](t)
        # second resblock
        t = self.stage[1](t)
        # the 3rd, 4th, and 5th residual blocks will output the feature layer
        out1 = self.stage[2](t)
        out2 = self.stage[3](out1)
        out3 = self.stage[4](out2)
        return out1, out2, out3

# Instantiate model
def darknet53(pretrained, **kwargs):
    # instantiate the model
    net = CSPDarkNet()
    # if it has been pre-trained (weights are stored)
    if pretrained:
        # load the pre-trained files directly
        if isinstance(pretrained, str):
            net.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return net




















