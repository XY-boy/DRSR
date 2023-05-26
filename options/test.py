import argparse

parser = argparse.ArgumentParser(description='DASR_test')
# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0, help='number of threads for data loading')
parser.add_argument('--cpu', type=bool, default=False, help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/data/XY_space/Datasets/JILIN_test/GT/', help='dataset directory')
parser.add_argument('--scale', type=str, default='4', help='super resolution scale')
parser.add_argument('--save_model_dir', type=str, default='checkpoint_2_E/', help='save model results')
parser.add_argument('--pretrained_name', type=str, default='4x_blindsr_epoch_695.pth', help='save model name')
parser.add_argument('--save_results_dir', type=str, default='Results/bs8_dual_m/iso/JILIN/', help='save output results')


# Degradation specifications
parser.add_argument('--blur_kernel', type=int, default=21, help='size of blur kernels=21, the same as DASR')
parser.add_argument('--blur_type', type=str, default='aniso_gaussian', help='blur types (iso_gaussian | aniso_gaussian)')
parser.add_argument('--mode', type=str, default='bicubic', help='downsampler (bicubic | s-fold)')
parser.add_argument('--noise', type=float, default=0.0, help='noise level')

# iso
parser.add_argument('--sig', type=float, default=0.2, help='specific sigma of isotropic Gaussian blurs')  # 测试的时候定义
# aniso
parser.add_argument('--lambda_1', type=float, default=0.2, help='one eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--lambda_2', type=float, default=4.0, help='another eigenvalue of anisotropic Gaussian blurs')
parser.add_argument('--theta', type=float, default=0.0, help='rotation angle of anisotropic Gaussian blurs [0, 180]')

args_test = parser.parse_args()