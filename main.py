import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from options.train import args_train
from options.test import args_test
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from load_data.data import get_training_set, get_eval_set
from torch.utils.data import DataLoader
from module.base_module import print_network, checkpoint, PSNR, save_best_model
from module.blindsr import BlindSR as Net
from SimCLR.nt_xent import NT_Xent
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
from tensorboardX import SummaryWriter
import numpy as np


gpus_list = range(args_train.n_GPUs)
hostname = str(socket.gethostname())
cudnn.benchmark = True
writer = SummaryWriter('runs/ours_w_realworld_training')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver', force=True)
    print('===> Loading datasets')
    train_set = get_training_set(args_train.dir_data, args_train.scale, args_train.patch_size,
                                 data_augmentation=True)
    training_data_loader = DataLoader(dataset=train_set, num_workers=args_train.n_threads,
                                      batch_size=args_train.batch_size, shuffle=True)

    eval_set = get_eval_set(args_train.data_val_while_training, args_train.scale)
    eval_data_loader = DataLoader(dataset=eval_set, num_workers=args_train.n_threads, batch_size=1, shuffle=False)

    print('===> Building model ')
    model = Net(args_train)
    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    model_E = torch.nn.DataParallel(model.module.E, device_ids=gpus_list)
    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')
    if args_train.pretrained_flag:
        model_name = os.path.join(args_train.save_model_dir + args_train.pretrained_name)
        print(model_name)
        if os.path.exists(model_name):
            state_dit = model.state_dict()
            weights = torch.load(model_name)
            model.load_state_dict(weights, strict=False)
            # model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
            print('pretrained model is loaded!')
        else:
            raise Exception("No such pretrained model!!c")
    # To GPU
    model = model.cuda(gpus_list[0])
    loss_sr = nn.L1Loss().cuda(gpus_list[0])  # L1 loss 约束超分过程
    loss_ntx = NT_Xent(batch_size=args_train.batch_size, temperature=0.1)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8)

    best_epoch = 0
    best_eval_psnr = 0.0
    for epoch in range(args_train.start_epoch, args_train.epochs_encoder + args_train.epochs_sr + 1):
        sr_loss = 0
        c_loss = 0
        model.train()

        # lr stepwise
        if epoch <= args_train.epochs_encoder:
            lr = args_train.lr_encoder * (args_train.gamma_encoder ** (epoch // args_train.lr_decay_encoder))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = args_train.lr_sr * (args_train.gamma_sr ** ((epoch - args_train.epochs_encoder) // args_train.lr_decay_sr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for iteration, batch in enumerate(training_data_loader, 1):
            gt_i, gt_j, lr_d_i, lr_d_j = batch[0], batch[1], batch[2], batch[3]

            gt_i = Variable(gt_i).cuda(gpus_list[0])  # [b 3 H W]
            gt_j = Variable(gt_j).cuda(gpus_list[0])  # [b 3 h w]
            lr_d_i = Variable(lr_d_i).cuda(gpus_list[0])  # [b 3 h/2 w/2], 以(h/2，w/2)在LR中随机裁剪，裁剪大小可自定义
            lr_d_j = Variable(lr_d_j).cuda(gpus_list[0])
            # print(lr_d_i.size()) torch.cuda.empty_cache()

            optimizer.zero_grad()
            if epoch <50:
                t0 = time.time()
                h_i, h_j, mlp_i, mlp_j = model_E(gt_i, gt_j)
                loss_constrast = loss_ntx(mlp_i, mlp_j)  # 计算对比损失需要使用通过mlp的两个patch
                loss = loss_constrast
                t1 = time.time()
                c_loss += loss.item()
                print("===> real Epoch[{}]({}/{}): Constast Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                                  len(
                                                                                                      training_data_loader),
                                                                                                  loss_constrast.item(),
                                                                                                  (t1 - t0)))
            if epoch>=50 and epoch < args_train.epochs_encoder:
                t0 = time.time()
                h_i, h_j, mlp_i, mlp_j = model_E(lr_d_i, lr_d_j)
                loss_constrast = loss_ntx(mlp_i, mlp_j)  # 计算对比损失需要使用通过mlp的两个patch
                loss = loss_constrast
                t1 = time.time()
                c_loss += loss.item()
                print("===> encoder Epoch[{}]({}/{}): Constast Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                         len(training_data_loader),
                                                                                         loss_constrast.item(),
                                                                                         (t1 - t0)))
            if epoch>=args_train.epochs_encoder:
                t0 = time.time()
                sr_i, mlp_i, mlp_j = model(lr_d_i, lr_d_j)
                loss_SR = loss_sr(sr_i, gt_i)
                loss_constrast = loss_ntx(mlp_i, mlp_j)

                loss = loss_SR + loss_constrast
                t1 = time.time()
                c_loss += loss_constrast
                sr_loss += loss.item()
                print("===> all Epoch[{}]({}/{}): Constast Loss: {:.4f} SR Loss: {:.4f}|| Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                                  len(training_data_loader),
                                                                                                  loss_constrast,
                                                                                                  loss_SR.item(),
                                                                                                  (t1 - t0)))
            loss.backward()
            optimizer.step()

        print("===> Epoch {} Complete: Avg. C_Loss: {:.4f} Avg. SR_Loss: {:.4f}".format(epoch, c_loss / len(training_data_loader),
                                                                                        sr_loss / len(training_data_loader)))
        writer.add_scalar('Avg. ConstrastLoss', c_loss / len(training_data_loader), epoch)
        if epoch > args_train.epochs_encoder:
            writer.add_scalar('Avg. SRLoss', sr_loss / len(training_data_loader), epoch)

        # test while training
        if epoch > args_train.epochs_encoder:
            count = 1
            avg_psnr_pre = 0.0
            avg_psnr_test = 0.0

            model.eval()
            for batch in eval_data_loader:
                gt, lr_d, _ = batch[0], batch[1], batch[2]
                with torch.no_grad():
                    gt = Variable(gt).cuda(gpus_list[0])
                    lr_d = Variable(lr_d).cuda(gpus_list[0])
                t0 = time.time()
                with torch.no_grad():
                    prediction = model(lr_d, lr_d)
                t1 = time.time()
                print('===>Processing: %s || Timer: %4f sec' % (str(count), (t1 - t0)))

                prediction = prediction.cpu()
                prediction = prediction.data[0].numpy().astype(np.float32)

                gt = gt.cpu()
                gt = gt.squeeze(0).numpy().astype(np.float32)

                psnr_pre = PSNR(prediction, gt)
                print(psnr_pre)
                avg_psnr_pre += psnr_pre
                avg_psnr_test = avg_psnr_pre / len(eval_data_loader)
                count += 1

            print("===> Epoch {} Complete: Avg. PSNR: {:.4f}".format(epoch, avg_psnr_pre / len(eval_data_loader)))
            if avg_psnr_test > best_eval_psnr:
                best_epoch = epoch
                best_eval_psnr = avg_psnr_test
            if epoch == (args_train.epochs_encoder + args_train.epochs_sr):
                print('Best_epoch:{:.4f},Best_psnr={:.6f}'.format(best_epoch, best_eval_psnr))
                save_best_model(best_epoch, args_train, model)

            writer.add_scalar('Avg. PSNR', avg_psnr_pre / len(eval_data_loader), epoch)

        # save moledl for each epoch
        checkpoint(epoch, args_train, model)



