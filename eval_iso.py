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
gpus_list = range(args_test.n_GPUs)
hostname = str(socket.gethostname())

print('===> Loading test datasets')
eval_set = get_eval_set(args_test.dir_data, args_test.scale)
eval_data_loader = DataLoader(dataset=eval_set, num_workers=args_test.n_threads, batch_size=1, shuffle=False)

print('===> Building model ')
model = Net(args_test)
print('---------- Networks architecture -------------')
print_network(model)

# lr = torch.randn(1,3,150,150)
# flop, _ = profile(model, inputs=(lr, lr, ))
# print(flop/1e9)
model_name = os.path.join(args_test.save_model_dir + args_test.pretrained_name)
if os.path.exists(model_name):
    #model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load(model_name, map_location=lambda storage, loc: storage).items()})
    print('pretrained model is loaded!')
else:
    raise Exception('No such pretrained model!')

# To GPU
model = model.cuda(gpus_list[0])

def eval():
    count = 1
    avg_psnr_predicted = 0.0
    total_time = 0
    model.eval()
    for batch in eval_data_loader:
        gt, lr_d, im_path = batch[0], batch[1], batch[2]

        with torch.no_grad():
            gt = Variable(gt).cuda(gpus_list[0])
            lr_d = Variable(lr_d).cuda(gpus_list[0])
            print(lr_d.size())
        t0 = time.time()
        with torch.no_grad():
            prediction = model(lr_d, lr_d)
        t1 = time.time()
        total_time = total_time + (t1 - t0)
        print('===>Processing: %s || Timer: %4f sec' % (str(count), (t1 - t0)))

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)

        lr_d = lr_d.cpu()
        lr_d = lr_d.squeeze(0).numpy().astype(np.float32)

        gt = gt.cpu()
        gt = gt.squeeze(0).numpy().astype(np.float32)
        # m_scale = 4
        # w = int(np.floor(gt.shape[1]/m_scale))
        # h = int(np.floor(gt.shape[0] / m_scale))
        # gt = gt[0:m_scale*h, 0:m_scale*w]
        # psnr_pre = PSNR(prediction, gt)
        # avg_psnr_predicted = avg_psnr_predicted + psnr_pre
        # print(psnr_pre)

        # save imges
        img_name = os.path.split(im_path[0])[1]
        lr_d_folder = args_test.save_results_dir + '/Iso/sig' + str(args_test.sig) + '/LRBlur/'
        sr_folder = args_test.save_results_dir + '/Iso/sig' + str(args_test.sig) + '/SR/'
        save_images(lr=lr_d, sr=prediction, lr_folder=lr_d_folder, sr_folder=sr_folder, im_name=img_name)
        count += 1

    print("PSNR_predicted=", avg_psnr_predicted / count)
    print(total_time)

##Eval Start!!!!
eval()



