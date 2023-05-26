from os.path import join
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
from SimCLR.transformation import TransformsSimCLR
from load_data.dataset import DatasetFromFolderEval, DatasetFromFolder
from options.train import args_train
from options.test import args_test

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation):
    #hr_dir = join(data_dir, 'GT')
    hr_dir = data_dir

    return DatasetFromFolder(args_train, hr_dir, patch_size, upscale_factor, data_augmentation,
                             transform=None)

def get_eval_set(lr_dir, upscale_factor):
    return DatasetFromFolderEval(args_test, lr_dir, upscale_factor, transform=None)
