import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


def lowlight(image_path):
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    EFINet = model.K_Estimate_net().cuda()
    EFINet.load_state_dict(torch.load('snapshots/BestEpoch.pth'))
    start = time.time()
    _, enhanced_image1 = EFINet(data_lowlight)
    _, enhanced_image2 = EFINet(enhanced_image1)
    _, enhanced_image3 = EFINet(enhanced_image2)

    end_time = (time.time() - start)
    print(end_time)
    image_path = image_path.replace('test_data', 'result')
    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))

    torchvision.utils.save_image(enhanced_image3, result_path)


if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        filePath = 'data/test_data/'

        file_list = os.listdir(filePath)

        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                print(image)
                lowlight(image)
