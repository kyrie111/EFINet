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
import Myloss
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
from utils.print_time import print_time
import csv
import math
# from torch.utils.tensorboard import SummaryWriter


# 定义参数初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # 随机初始化采用正态分布，均值为0，标准差为0.02.
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)  # 将偏差定义为常量0


def train(config):
    train_dark_path = config.data_path + 'fivek_sp/mini_train_dark_gf5/'
    val_dark_path = config.data_path + 'fivek_sp/mini_val_dark_gf5/'
    train_gth_path = config.data_path + 'fivek_sp/mini_train_gth/'
    val_gth_path = config.data_path + 'fivek_sp/mini_val_gth/'

    # val_result_path = 'data_val/'
    # if not os.path.exists(val_result_path):
    #     os.makedirs(val_result_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # tensorboard
    # tb = SummaryWriter()

    EFINet = model.Enhancement_Fusion_Iterative_Net().cuda()

    EFINet.apply(weights_init)

    # if pretrained model is existed
    if config.load_pretrain:
        print('加载预训练模型')
        EFINet.load_state_dict(torch.load(config.pretrain_dir))

    # 计算模型参数数量
    total_params = sum(p.numel() for p in EFINet.parameters() if p.requires_grad)
    print("Total_params: {}".format(total_params))

    # 创建图像数据加载器
    transform = transforms.Compose([transforms.ToTensor()])
    train_path_list = [train_dark_path, train_gth_path]
    val_path_list = [val_dark_path, val_gth_path]
    train_data = dataloader.DataSet_loader(transform, config.gth_train, train_path_list)
    val_data = dataloader.DataSet_loader(transform, config.gth_train, val_path_list)
    train_data_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True,
                                   num_workers=config.num_workers, pin_memory=False)
    val_data_loader = DataLoader(val_data, batch_size=config.train_batch_size, shuffle=True,
                                 num_workers=config.num_workers, pin_memory=False)

    L_mae = Myloss.MAE().cuda()
    L_color = Myloss.color_loss().cuda()
    L_ssim = Myloss.SSIM().cuda()
    L_vgg = Myloss.VGG_LOSS().cuda()

    # 定义优化器
    optimizer = torch.optim.Adam(EFINet.parameters(), lr=config.start_lr, weight_decay=config.weight_decay)

    # warm_up_epochs = 5
    # warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / (warm_up_epochs) if epoch < warm_up_epochs \
    #     else 0.5 * (math.cos((epoch - warm_up_epochs) / (config.num_epochs - warm_up_epochs) * math.pi) + 1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, min_lr=0.0001)

    # get some random training images
    # dataiter = iter(train_data_loader)
    # i_name, i_dark, i_gth = dataiter.next()

    # create grid of images
    # dark_img_grid = torchvision.utils.make_grid(i_dark)
    # gth_img_grid = torchvision.utils.make_grid(i_gth)

    # write to tensorboard
    # tb.add_image('img_input', dark_img_grid)
    # tb.add_image('img_gth', gth_img_grid)

    # 保存训练trainloss
    f1 = open('trainLossLog.csv', 'w', encoding='utf-8', newline='' "")
    csv_writer1 = csv.writer(f1)
    csv_writer1.writerow(["loss_mae", "loss_color", "loss_ssim", "loss_vgg", "total_loss"])
    # 保存训练trainloss
    f2 = open('valLossLog.csv', 'w', encoding='utf-8', newline='' "")
    csv_writer2 = csv.writer(f2)
    csv_writer2.writerow(["loss_mae", "loss_color", "loss_ssim", "loss_vgg", "total_loss"])

    EFINet.train()
    # start training
    min_loss = 99999
    start_time = time.time()
    print("\nstart to train!")
    for epoch in range(config.num_epochs):
        index = 0
        train_loss = 0
        # train
        for name, dark_image, gt_image in train_data_loader:
            index = index + 1

            if index == 1:
                for i in range(3):
                    n = 'E%d_%s.jpg' % (epoch + 1, name[i])
                    torchvision.utils.save_image(dark_image[i], 'dark_input/' + n)

            if index == 1:
                for i in range(3):
                    n = 'E%d_%s.jpg' % (epoch + 1, name[i])
                    torchvision.utils.save_image(gt_image[i], 'gt_input/' + n)

            dark_image = dark_image.cuda()
            gt_image = gt_image.cuda()

            # 3 iteration
            Initial_enhanced_image1, enhanced_image1 = EFINet(dark_image)
            Initial_enhanced_image2, enhanced_image2 = EFINet(enhanced_image1)
            Initial_enhanced_image3, enhanced_image3 = EFINet(enhanced_image2)
            enhanced_image = [enhanced_image1, enhanced_image2, enhanced_image3]

            if index == 1:
                for i in range(3):
                    n = 'E%d_%s.jpg' % (epoch + 1, name[i])
                    torchvision.utils.save_image(enhanced_image3[i], 'train_test/' + n)

            loss_mae = L_mae(enhanced_image3, gt_image)
            loss_color = L_color(enhanced_image3, gt_image)
            loss_ssim = L_ssim(enhanced_image3, gt_image)
            loss_vgg = L_vgg(enhanced_image3, gt_image)

            # train loss
            loss = 3 * loss_color + loss_mae + loss_ssim + loss_vgg
            # train_loss = train_loss + loss.item()

            if np.mod(index, config.itr_to_excel) == 0:
                print('epoch %d, %03d/%d, loss: %f' % (epoch + 1, index, len(train_data_loader), loss.item()))
                csv_writer1.writerow(
                    [loss_mae.item(), loss_color.item(), loss_ssim.item(), loss_vgg.item(), loss.item()])
                print_time(start_time, index, config.num_epochs, len(train_data_loader), epoch)
                # create grid of images
                train_result_grid = torchvision.utils.make_grid(enhanced_image3)
                # write to tensorboard
                # tb.add_image('train_result', train_result_grid)
                # tb.add_scalar('image training loss', loss, epoch * len(train_data_loader) + index)
            
            loss = loss / config.accumulation_steps
            loss.backward()

            if ((index + 1) % config.accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

            if ((index + 1) % config.snapshot_iter) == 0:
                torch.save(EFINet.state_dict(), config.snapshots_folder + "Epoch" + str(epoch + 1) + '.pth')

        # train_loss = train_loss / len(train_data_loader)
        optimizer.zero_grad()
        optimizer.step()
        # print("the lr of %d epoch:%f" % (epoch, optimizer.param_groups[0]['lr']))
        # scheduler.step()
        # print("the new_lr of %d epoch:%f" % (epoch, optimizer.param_groups[0]['lr']))

        # eval
        val_loss = 0
        with torch.no_grad():
            EFINet.eval()
            for name, dark_image, gt_image in val_data_loader:
                dark_image = dark_image.cuda()
                gt_image = gt_image.cuda()

                # 3 iteration
                Initial_enhanced_image1, enhanced_image1 = EFINet(dark_image)
                Initial_enhanced_image2, enhanced_image2 = EFINet(enhanced_image1)
                Initial_enhanced_image3, enhanced_image3 = EFINet(enhanced_image2)
                # enhanced_image = [enhanced_image1, enhanced_image2, enhanced_image3]

                loss_mae = L_mae(enhanced_image3, gt_image)
                loss_color = L_color(enhanced_image3, gt_image)
                loss_ssim = L_ssim(enhanced_image3, gt_image)
                loss_vgg = L_vgg(enhanced_image3, gt_image)

                # val loss
                loss = 3 * loss_color + loss_mae + loss_ssim + loss_vgg

                if np.mod(index, config.itr_to_excel) == 0:
                    csv_writer2.writerow(
                        [loss_mae.item(), loss_color.item(), loss_ssim.item(), loss_vgg.item(), loss.item()])
                    for n in name:
                        torchvision.utils.save_image(enhanced_image3, val_result_path + n + '_' + str(epoch) + '.jpg')
                    print('val loss = {}\n'.format(loss.item()))

                val_loss = val_loss + loss.item()

        val_loss = val_loss / len(val_data_loader)
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(EFINet.state_dict(), config.snapshots_folder + 'BestEpoch.pth')
            print('saving the best epoch %d model with %.5f' % (epoch + 1, min_loss))

    f1.close()
    f2.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--data_path', type=str, default="/home/liu/wufanding/data/")
    parser.add_argument('--start_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/BestEpoch.pth")
    parser.add_argument('--gth_train', help='Whether to add Gth training', default=False, type=bool)
    parser.add_argument('--accumulation_steps', help='Set the accumulation steps', default=8, type=int)
    parser.add_argument('--itr_to_excel', help='Save to excel after every n trainings', default=64, type=int)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
