import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# Convolution operation
class ConLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseCon2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DenseCon2d, self).__init__()
        self.dense_conv = ConLayer(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseCon2d(in_channels, out_channels_def, kernel_size, stride, padding),
                       DenseCon2d(in_channels + out_channels_def, out_channels_def, kernel_size, stride, padding),
                       DenseCon2d(in_channels + out_channels_def * 2, out_channels_def, kernel_size, stride, padding)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class Enhancement_Fusion_Iterative_Net(nn.Module):

    def __init__(self):
        super(Enhancement_Fusion_Iterative_Net, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # Number of channels in the middle hidden layer
        number_f = 16

        # k-map estimate net
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 3, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # fusion net
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16, 128]
        kernel_size = 3
        stride = 1
        padding = 1

        # encoder
        self.f_conv1 = ConLayer(3, nb_filter[0], kernel_size, stride)
        self.f_DB1 = denseblock(nb_filter[0], kernel_size, stride, padding)

        # decoder
        self.f_conv2 = ConLayer(nb_filter[4], nb_filter[1], kernel_size, stride)
        self.f_conv3 = ConLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.f_conv4 = ConLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.f_conv5 = ConLayer(nb_filter[3], 3, kernel_size, stride)

    def forward(self, x):
        # k-map estimate net
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = self.e_conv7(torch.cat([x1, x6], 1))

        Initial_enhanced_image = x * x_r

        # fusion net
        # encoder
        x11 = self.f_conv1(x)
        x12 = self.f_DB1(x11)
        x21 = self.f_conv1(Initial_enhanced_image)
        x22 = self.f_DB1(x21)
        # fusion
        f_0 = torch.cat([x12, x22], 1)
        # decoder
        x3 = self.f_conv2(f_0)
        x4 = self.f_conv3(x3)
        x5 = self.f_conv4(x4)
        output = self.f_conv5(x5)

        return Initial_enhanced_image, output
