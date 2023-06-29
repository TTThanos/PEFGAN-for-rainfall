import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import time
from os import mkdir
from os.path import join, isdir

from torch.utils.data import random_split, DataLoader

from datetime import date
from torchvision import transforms

from utils import Huber, ACCESS_AWAP_GAN, RMSE

import cv2

import os
import random
from datetime import date, timedelta

import torch
import torch.nn as nn

import time
import numpy as np
import xarray as xr
import cv2
from torch.utils.data import Dataset

### USER PARAMS ###
START_TIME = date(1990, 1, 1)
END_TIME = date(2001, 12, 31)

EXP_NAME = "DESRGAN"
VERSION = "3"
UPSCALE = 8  # upscaling factor

NB_BATCH = 3  # mini-batch
NB_Iteration = 10
PATCH_SIZE = 576  # Training patch size
NB_THREADS = 36

START_ITER = 0  # Set 0 for from scratch, else will load saved params and trains further


L_ADV = 1e-3  # Scaling params for the Adv loss
L_FM = 1  # Scaling params for the feature matching loss
L_LPIPS = 1e-3  # Scaling params for the LPIPS loss

LR_G = 1e-5  # Learning rate for the generator
LR_D = 1e-5  # Learning rate for the discriminator

best_avg_rmses = 1


def write_log(log):
    print(log)
    if not os.path.exists("./save/"):
        os.mkdir("./save/")
    if not os.path.exists("./save/" + VERSION + "/"):
        os.mkdir("./save/" + VERSION + "/")
    my_log_file = open("./save/" + VERSION + '/train.txt', 'a')
    #     log="Train for batch %d,data loading time cost %f s"%(batch,start-time.time())
    my_log_file.write(log + '\n')
    my_log_file.close()
    return


### Generator ###
## ESRGAN for x8
import RRDBNet_arch as arch

model_G = arch.RRDBNetx4x2(1, 1, 64, 23, gc=32).cuda()

if torch.cuda.device_count() > 1:
    write_log("!!!Let's use" + str(torch.cuda.device_count()) + "GPUs!")
    model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))


### U-Net Discriminator ###
# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, wide=True,
                 preactivation=True, activation=nn.LeakyReLU(0.1, inplace=False), downsample=nn.AvgPool2d(2, stride=2)):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

        self.bn1 = self.which_bn(self.hidden_channels)
        self.bn2 = self.which_bn(out_channels)

    # def shortcut(self, x):
    #     if self.preactivation:
    #         if self.learnable_sc:
    #             x = self.conv_sc(x)
    #         if self.downsample:
    #             x = self.downsample(x)
    #     else:
    #         if self.downsample:
    #             x = self.downsample(x)
    #         if self.learnable_sc:
    #             x = self.conv_sc(x)
    #     return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = self.activation(x)
        else:
            h = x
        h = self.bn1(self.conv1(h))
        # h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h  # + self.shortcut(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, activation=nn.LeakyReLU(0.1, inplace=False),
                 upsample=nn.Upsample(scale_factor=2, mode='nearest')):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(out_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            # x = self.upsample(x)
        h = self.bn1(self.conv1(h))
        # h = self.activation(self.bn2(h))
        # h = self.conv2(h)
        # if self.learnable_sc:
        #     x = self.conv_sc(x)
        return h  # + x


class UnetD(torch.nn.Module):
    def __init__(self):
        super(UnetD, self).__init__()

        self.enc_b1 = DBlock(1, 64, preactivation=False)
        self.enc_b2 = DBlock(64, 128)
        self.enc_b3 = DBlock(128, 192)
        self.enc_b4 = DBlock(192, 256)
        self.enc_b5 = DBlock(256, 320)
        self.enc_b6 = DBlock(320, 384)
        # 这里320，384是否与图像大小有关？
        self.enc_out = nn.Conv2d(384, 1, kernel_size=1, padding=0)

        self.dec_b1 = GBlock(384, 320)
        self.dec_b2 = GBlock(320 * 2, 256)
        self.dec_b3 = GBlock(256 * 2, 192)
        self.dec_b4 = GBlock(192 * 2, 128)
        self.dec_b5 = GBlock(128 * 2, 64)
        self.dec_b6 = GBlock(64 * 2, 32)

        self.dec_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # print(classname)
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        e1 = self.enc_b1(x)
        e2 = self.enc_b2(e1)
        e3 = self.enc_b3(e2)
        e4 = self.enc_b4(e3)
        e5 = self.enc_b5(e4)
        e6 = self.enc_b6(e5)
        e_out = self.enc_out(F.leaky_relu(e6, 0.1))

        d1 = self.dec_b1(e6)
        d2 = self.dec_b2(torch.cat([d1, e5], 1))
        d3 = self.dec_b3(torch.cat([d2, e4], 1))
        d4 = self.dec_b4(torch.cat([d3, e3], 1))
        d5 = self.dec_b5(torch.cat([d4, e2], 1))
        d6 = self.dec_b6(torch.cat([d5, e1], 1))

        d_out = self.dec_out(F.leaky_relu(d6, 0.1))

        return e_out, d_out, [e1, e2, e3, e4, e5, e6], [d1, d2, d3, d4, d5, d6]


model_D = UnetD().cuda()

## Optimizers
params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
opt_G = optim.Adam(params_G, lr=LR_G)

params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
opt_D = optim.Adam(params_D, lr=LR_D)

## Load saved params
if START_ITER > 0:
    lm = torch.load('{}/checkpoint/v{}/model_G_i{:06d}.pth'.format(EXP_NAME, str(VERSION), START_ITER))
    model_G.load_state_dict(lm.state_dict(), strict=True)

lr_transforms = transforms.Compose([
    transforms.ToTensor()
])

hr_transforms = transforms.Compose([
    transforms.ToTensor()
])
def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]



def get_filename_with_time_order(rootdir,START_TIME= START_TIME, END_TIME = END_TIME ):
    '''
    get filename first and generate label ,one different w
    Check whether the date in other ensemble folders exist in e09.
    If the date exists in e09 folder, then add the corresponding data into filename_list.

    return: e_num, data, AWAP_date, leading time
    '''
    ensemble = ['e01', 'e02', 'e03', 'e04', 'e05', 'e06', 'e07', 'e08', 'e09']
    dates = date_range(START_TIME, END_TIME)
    leading_time_we_use = 7

    _files = []
    for en in ensemble:
        for date in dates:
            access_path = rootdir + "e09" + "/da_pr_" + date.strftime("%Y%m%d") + "_" + "e09" + ".nc"
            #                 print(access_path)
            if os.path.exists(access_path):
                for i in range(leading_time_we_use):
                    if date == END_TIME and i == 1:
                        break
                    path = [en]
                    AWAP_date = date + timedelta(i)
                    path.append(date)
                    path.append(AWAP_date)
                    path.append(i)
                    _files.append(path)

    # 最后去掉第一行，然后shuffle
    return _files
file_ACCESS_dir = "/scratch/iu60/yl3101/Processed_data/"
filename_list = get_filename_with_time_order(file_ACCESS_dir)
print(filename_list)