import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from options.test import args_test
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from load_data.data import get_training_set, get_eval_set
from torch.utils.data import DataLoader
from module.base_module import PSNR, save_images, print_network
from module.blindsr import BlindSR as Net
from SimCLR.my_nt_xent import NT_Xent
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
from tensorboardX import SummaryWriter
import numpy as np
import cv2
from thop import profile
import glob
import imageio

gpus_list = range(args_test.n_GPUs)
hostname = str(socket.gethostname())

def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor

print('===> Loading test datasets')
noise = '/'
lr_folder = '/data/XY_space/Datasets/DOTA_test/GT/' + noise
model_name = 'checkpoint_woSimclr/4x_blindsr_epoch_554.pth'
save_results_dir = 'Results/bs8_dual_m/real/AID2'

print('===> Building model ')
model = Net(args_test)
print('---------- Networks architecture -------------')
print_network(model)

# model_name = os.path.join(args_test.save_model_dir + args_test.pretrained_name)
if os.path.exists(model_name):
    #model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load(model_name, map_location=lambda storage, loc: storage).items()})
    print('pretrained model is loaded!')
else:
    raise Exception('No such pretrained model!')

# To GPU
model = model.cuda(gpus_list[0])

def eval(theta):
    count = 1
    print(theta)
    total_time = 0
    model.eval()
    lr_list = lr_folder + theta + '*.png'
    lr_imgs = glob.glob(lr_list)

    for i in range(lr_imgs.__len__()):
        lr = imageio.imread(lr_imgs[i])
        lr = np2Tensor(lr)
        lr = lr.unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            lr = Variable(lr).cuda(gpus_list[0])
            prediction = model(lr, lr)
        t1 = time.time()
        total_time = total_time + (t1 - t0)
        print('===>Processing: %s || Timer: %4f sec' % (str(count), (t1 - t0)))

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)

        lr = lr.cpu()
        lr = lr.squeeze(0).numpy().astype(np.float32)
        # save imges
        img_name = os.path.split(lr_imgs[i])[1]
        lr_d_folder = save_results_dir + noise + theta + '/LRBlur/'
        sr_folder = save_results_dir + noise + theta + ''
        save_images(lr=lr, sr=prediction, lr_folder=lr_d_folder, sr_folder=sr_folder, im_name=img_name)
        count += 1

    # print("PSNR_predicted=", avg_psnr_predicted / count)
    print(total_time)

##Eval Start!!!!
# theta_list = ['theta0/', 'theta0.1/', 'theta0.2/', 'theta0.3/', 'theta0.4/', 'theta0.5/', 'theta0.6/', 'theta0.7/', 'theta0.8/',
#               'theta0.9/', 'theta1/']
# for theta in theta_list:
theta=''
eval(theta=theta)



