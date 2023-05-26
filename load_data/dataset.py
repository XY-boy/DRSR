import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
from utils import util
from torchvision.transforms import Compose, ToTensor
import imageio

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = imageio.imread(filepath)
    # img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img, patch_size=48, scale=1):
    th, tw = img.shape[:2]  ## HR image

    tp = round(scale * patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :]


def augment(img, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    if hflip:
        img = img[:, ::-1, :]
    if vflip:
        img = img[::-1, :, :]
    if rot90:
        img = img.transpose(1, 0, 2)

    return img

def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor
"""
        # sig, sig_min and sig_max are used for isotropic Gaussian blurs
        During training phase (random=True):
            the width of the blur kernel is randomly selected from [sig_min, sig_max]
        During test phase (random=False):
            the width of the blur kernel is set to sig

        # lambda_1, lambda_2, theta, lambda_min and lambda_max are used for anisotropic Gaussian blurs
        During training phase (random=True):
            the eigenvalues of the covariance is randomly selected from [lambda_min, lambda_max]
            the angle value is randomly selected from [0, pi]
        During test phase (random=False):
            the eigenvalues of the covariance are set to lambda_1 and lambda_2
            the angle value is set to theta
"""
class DatasetFromFolder(data.Dataset):  # 读取一张HR，按patch返回HR，lr,退化模糊核b_kernel以及lr的两个正样本lr_i，lr_j
    def __init__(self, args, HR_dir, patch_size, upscale_factor, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.args = args
        self.patch_size = patch_size
        self.upscale_factor = int(upscale_factor)
        self.transform = transform
        self.data_augmentation = data_augmentation

        self.degrade = util.SRMDPreprocessing(
            self.upscale_factor,
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

    def __getitem__(self, index):
        target = load_img(self.hr_image_filenames[index])  # 读取图片[600 600 3]
        patch_i = get_patch(target, self.patch_size)  # 随机裁剪patch, numpy:[h w 3]
        patch_j = get_patch(target, self.patch_size)

        if self.data_augmentation:  # 旋转翻转数据扩充
            patch_i = augment(patch_i)
            patch_j = augment(patch_j)

        patch_i = np2Tensor(patch_i).unsqueeze(0).cuda()  # (h w 3)numpy image要先转成tensor[1, 2, 3, h, w](rgbrange = 255)
        patch_j = np2Tensor(patch_j).unsqueeze(0).cuda()  # [1, 3, h, w]
        patch_i_j = torch.stack([patch_i, patch_j], dim=1)  # [1, 2, 3, h, w]

        input_d_i_j, bkernel = self.degrade(patch_i_j, random=True)  # degradation

        input_d_i = input_d_i_j[:, 0, :, :, :].squeeze(0)
        input_d_j = input_d_i_j[:, 1, :, :, :].squeeze(0)  # [3 h w]

        patch_i_gt = patch_i.squeeze(0)
        patch_j_gt = patch_j.squeeze(0)  # [3 H W]

        # if self.transform:  # 在LR中随机裁剪LR的size//2（可自定义） + 随机旋转, 返回两个数据增强的inputi，j
        #     input_i, input_j = self.transform(input_d)

        return patch_i_gt, patch_j_gt, input_d_i, input_d_j

    def __len__(self):
        return len(self.hr_image_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, args_test, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = int(upscale_factor)
        self.transform = transform
        self.args = args_test

        self.degrade = util.SRMDPreprocessing(
                    self.upscale_factor,
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=self.args.sig,
                    lambda_1=self.args.lambda_1,
                    lambda_2=self.args.lambda_2,
                    theta=self.args.theta,
                    noise=self.args.noise
                )

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        target = np2Tensor(target).unsqueeze(0).unsqueeze(0).cuda()  # 退化之前PIL image要先转成tensor[1, 1, 3, h, w](rgbrange = 255)
        input_d, bkernel = self.degrade(target, random=False)  # degradation,random=False代表测试的时候自己定义模糊核了，不再是训练时的随机模糊

        input_d = input_d.squeeze(0).squeeze(0)  # [3 H W]
        target = target.squeeze(0).squeeze(0)  # [3 h w]


        return target, input_d, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)