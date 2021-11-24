import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class MS_SSIM(torch.nn.Module):
    def __init__(self, size_average=True, max_val=255, channel=3):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = channel
        self.max_val = max_val

    def _ssim(self, img1, img2, size_average=True):
        # 读取图像的长宽通道
        _, c, w, h = img1.size()
        # 取wh11中的最小值，卷积核溢出
        window_size = min(w, h, 11)
        # 制作高斯核的sigma值
        sigma = 1.5 * window_size / 11
        # 制作高斯卷积核
        window = create_window(window_size, sigma, self.channel).cuda()

        # 对图1 和图2进行卷积
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
        mcs_map = V1 / V2
        if size_average:
            # return ssim_map.mean(), mcs_map.mean()
            return ssim_map.mean()

    def ms_ssim(self, img1, img2, levels=5):
        # 创建了一个一维的张量。
        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

        # 创建了两个第一维度为5的张量
        msssim = Variable(torch.Tensor(levels, ).cuda())
        mcs = Variable(torch.Tensor(levels, ).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            # 这里传输不同尺度的ssim图
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            # 这里下采样
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            # 下采样以后重新赋给原值
            img1 = filtered_im1
            img2 = filtered_im2
        # torch.prod返回张量上所有元素的积
        value = (torch.prod(mcs[0:levels - 1] ** weight[0:levels - 1]) *
                 (msssim[levels - 1] ** weight[levels - 1]))
        # print("value=%f" % value)
        return value

    def forward(self, img1, img2):

        return self._ssim(img1, img2)
