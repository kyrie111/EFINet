import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
from torchvision import models
from ms_ssim import *
import time
from torch.autograd import Variable
import numpy as np
from math import exp
import cv2
import torchvision.transforms as transformer


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, x1, x2):
        x1 = x1 - x2
        x1 = abs(x1)
        return x1.mean()


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x1, x2):
        x1 = x1 - x2
        x1 = x1.pow(2)
        return x1.mean()


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
        return out


class VGG_LOSS(nn.Module):
    def __init__(self):
        super(VGG_LOSS, self).__init__()
        # self.vgg = Vgg16().type(torch.cuda.FloatTensor)
        self.vgg = VGG16()
        self.l2 = MSE()

    def forward(self, dark, gth):
        output_features = self.vgg(dark)
        gth_features = self.vgg(gth)
        out_1 = self.l2(output_features[0], gth_features[0])
        out_2 = self.l2(output_features[1], gth_features[1])
        out_3 = self.l2(output_features[2], gth_features[2])
        out_4 = self.l2(output_features[3], gth_features[3])
        return (out_1 + out_2 + out_3 + out_4) / 4


class color_loss(nn.Module):
    def __init__(self):
        super(color_loss, self).__init__()

    def forward(self, x, y):
        b, c, h, w = x.shape

        mr_x, mg_x, mb_x = torch.split(x, 1, dim=1)
        mr_x, mg_x, mb_x = mr_x.view([b, 1, -1, 1]), mg_x.view([b, 1, -1, 1]), mb_x.view([b, 1, -1, 1])
        xx = torch.cat([mr_x, mg_x, mb_x], dim=3).squeeze(1) + 0.000001

        mr_y, mg_y, mb_y = torch.split(y, 1, dim=1)
        mr_y, mg_y, mb_y = mr_y.view([b, 1, -1, 1]), mg_y.view([b, 1, -1, 1]), mb_y.view([b, 1, -1, 1])
        yy = torch.cat([mr_y, mg_y, mb_y], dim=3).squeeze(1) + 0.000001

        xx = xx.reshape(h * w * b, 3)
        yy = yy.reshape(h * w * b, 3)
        l_x = torch.sqrt(pow(xx[:, 0], 2) + pow(xx[:, 1], 2) + pow(xx[:, 2], 2))
        l_y = torch.sqrt(pow(yy[:, 0], 2) + pow(yy[:, 1], 2) + pow(yy[:, 2], 2))
        xy = xx[:, 0] * yy[:, 0] + xx[:, 1] * yy[:, 1] + xx[:, 2] * yy[:, 2]
        cos_angle = xy / (l_x * l_y + 0.000001)
        angle = torch.acos(cos_angle.cpu())
        angle2 = angle * 360 / 2 / np.pi
        # re = angle2.reshape(b, -1)
        an_mean = torch.mean(angle2) / 100
        return an_mean.cuda()


class SSIM(nn.Module):
    def __init__(self, max_val=255, channel=3):
        super(SSIM, self).__init__()
        self.channel = channel
        self.max_val = max_val

    def ssim(self, img1, img2):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=self.channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=self.channel) - mu1_mu2
        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        # mcs_map = V1 / V2
        return ssim_map.mean()

    def forward(self, img1, img2):
        ssim = self.ssim(img1, img2)
        return 1 - ssim