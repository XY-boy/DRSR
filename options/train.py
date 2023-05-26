import argparse


parser = argparse.ArgumentParser(description='DASR')
# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--cpu', type=bool, default=False, help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# 训练集、测试集、patchsize、scale_factor
parser.add_argument('--dir_data', type=str, default='/data/XY_space/Datasets/AID_train/Isotropic/training/HR/x4/',
                    help='dataset directory')
parser.add_argument('--data_val_while_training', type=str, default='/data/XY_space/Datasets/AID_test/Isotropic/sig0.2/HR/x4/',
                    help='test dataset name')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--scale', type=str, default='4', help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=200, help='intput patch size')
parser.add_argument('--epochs_encoder', type=int, default=100, help='number of epochs to train the degradation encoder')
parser.add_argument('--epochs_sr', type=int, default=600, help='number of epochs to train the whole network')
parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate to train the degradation encoder')
parser.add_argument('--lr_sr', type=float, default=1e-4, help='learning rate to train the whole network')
parser.add_argument('--lr_decay_encoder', type=int, default=60, help='learning rate decay per N epochs')
parser.add_argument('--lr_decay_sr', type=int, default=150, help='learning rate decay per N epochs')
parser.add_argument('--gamma_encoder', type=float, default=0.1, help='learning rate decay factor for step decay')
parser.add_argument('--gamma_sr', type=float, default=0.5, help='learning rate decay factor for step decay')

parser.add_argument('--start_epoch', type=int, default=0, help='resume from the snapshot, and the start_epoch')
parser.add_argument('--save_flag', type=str, default='blindsr', help='file name to save')
parser.add_argument('--save_model_dir', type=str, default='checkpoint_woSimclr/', help='save output results')

parser.add_argument('--pretrained_flag', type=bool, default=False, help='load pretrained model or not')
parser.add_argument('--pretrained_name', type=str, default='4x_blindsr_epoch_100.pth', help='save output results')

# Degradation specifications
parser.add_argument('--blur_kernel', type=int, default=21, help='size of blur kernels=21, the same as DASR')
parser.add_argument('--blur_type', type=str, default='aniso_gaussian',
                    help='blur types (iso_gaussian | aniso_gaussian)')
parser.add_argument('--mode', type=str, default='bicubic',
                    help='downsampler (bicubic | s-fold)')
parser.add_argument('--noise', type=float, default=25, help='noise level')

## isotropic Gaussian blur
parser.add_argument('--sig_min', type=float, default=0.2, help='minimum sigma of isotropic Gaussian blurs')
parser.add_argument('--sig_max', type=float, default=4.0, help='maximum sigma of isotropic Gaussian blurs')

parser.add_argument('--sig', type=float, default=3.4, help='specific sigma of isotropic Gaussian blurs')  # 测试的时候定义


## anisotropic Gaussian blur
parser.add_argument('--lambda_min', type=float, default=0.2, help='minimum value for the eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--lambda_max', type=float, default=4.0, help='maximum value for the eigenvalue of anisotropic Gaussian blurs')

# --------------- 测试的时候定义下面的三个参数 -------------------
parser.add_argument('--lambda_1', type=float, default=0.2, help='one eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--lambda_2', type=float, default=4.0, help='another eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--theta', type=float, default=0.0, help='rotation angle of anisotropic Gaussian blurs [0, 180]')

args_train = parser.parse_args()