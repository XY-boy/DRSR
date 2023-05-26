import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))

def checkpoint(epoch, args, model):
    model_out_path = args.save_model_dir + str(
        args.scale) + 'x_' + args.save_flag + "_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
def save_best_model(bestepoch, args, model):
    model_out_path = args.save_model_dir + 'best_' + str(
        args.scale) + 'x_' + args.save_flag + "_epoch_{}.pth".format(bestepoch)
    torch.save(model.state_dict(), model_out_path)
    print("BestModel saved to {}".format(model_out_path))

def PSNR(pred, gt, shave_border=0):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def save_images(lr, sr, lr_folder, sr_folder, im_name):
    save_img_lr = lr.transpose(1, 2, 0)
    save_img_sr = sr.transpose(1, 2, 0)

    save_lr_dir = os.path.join(lr_folder)
    save_sr_dir = os.path.join(sr_folder)

    if not os.path.exists(save_lr_dir):
        os.makedirs(save_lr_dir)
    if not os.path.exists(save_sr_dir):
        os.makedirs(save_sr_dir)

    save_lr_fn = save_lr_dir + im_name
    save_sr_fn = save_sr_dir + im_name
    cv2.imwrite(save_lr_fn, cv2.cvtColor(save_img_lr, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(save_sr_fn, cv2.cvtColor(save_img_sr, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #print(save_fn)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
