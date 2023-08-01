import os
import sys
import cv2
import numpy as np
import torch

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
    from utils import util
except ImportError:
    pass


def generate_mod_LR_bic(lamuda1, lamuda2, thet):
    # set parameters
    up_scale = 4
    mod_scale = 4
    # set data dir
    sourcedir = 'D:\Real_world_SR\Data_preparation\AID_test\GT' #'/mnt/yjchai/SR_data/DIV2K_test_HR' #'/mnt/yjchai/SR_data/Flickr2K/Flickr2K_HR'
    savedir = 'D:\Real_world_SR\Data_preparation\AID_test/aniso/noise_15/theta'+str(thet) #'/mnt/yjchai/SR_data/DIV2K_test' #'/mnt/yjchai/SR_data/Flickr2K_train'
    # set random seed
    util.set_random_seed(0)

    # load PCA matrix of enough kernel
    print('load PCA matrix')
    pca_matrix = torch.load('D:\Github-package\Real-world-image-SR\Baseline\IKC-master\codes/pca_matrix.pth', map_location=lambda storage, loc: storage)
    print('PCA matrix shape: {}'.format(pca_matrix.shape))


    saveLRblurpath = os.path.join(savedir)

    if not os.path.exists(saveLRblurpath):
        os.makedirs(saveLRblurpath)
    else:
        print('It will cover '+ str(saveLRblurpath))

    filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith('.png')])
    print(filepaths)
    num_files = len(filepaths)

    kernel_map_tensor = torch.zeros((num_files, 1, 10)) # each kernel map: 1*10 ，21*21的模糊核被PCA编码成了1*10的向量

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]

        # LR_blur, by random gaussian kernel
        img_HR = util.img2tensor(image_HR)
        C, H, W = img_HR.size()
        # sig_list = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
        sig = 0.2  # 用于制作stable iso

        # 用来制作stable aniso
        lamda1 = lamuda1
        lamda2 = lamuda2
        theta = thet

        # 制作训练集random=true，sig∈[0.2,4.0]；制作测试集的时候random=false，也就是sig固定
        # rate_iso=1，只考虑各向同性，rate_iso=0，只考虑各向异性
        # noise_high代表最大噪声，用来构建随机噪声训练集，设置为25/255=0.098，noise_stable用来构建测试集，水平为
        # 0 5 10 的时候，noise_stable = 0 5/255 10/255
        prepro = util.SRMDPreprocessing(up_scale, pca_matrix, random=False, para_input=10, kernel=21, noise=False,
                                        cuda=True, sig=sig, sig_min=0.2, sig_max=4.0, rate_iso=0, scaling=3,
                                        rate_cln=0.2, noise_high=25/255, noise_stable=15/255, lam1=lamda1, lam2=lamda2, theta=theta)

        LR_img, ker_map = prepro(img_HR.view(1, C, H, W))
        image_LR_blur = util.tensor2img(LR_img)
        cv2.imwrite(os.path.join(saveLRblurpath, filename), image_LR_blur)  # 训练集由于sig是随机的，所以不用sig命名了，直接原始名字

        kernel_map_tensor[i] = ker_map
        torch.cuda.empty_cache()
    # save dataset corresponding kernel maps
    # torch.save(kernel_map_tensor, './AID_train_kermap_Aniso_G.pth')
    print("Image Blurring & Down smaple Done: X"+str(up_scale))

if __name__ == "__main__":
    dagradation_list = [[2.0, 0.6, 0], [1.8, 1.3, 0.1], [2.6, 1.6, 0.2], [3.8, 1.4, 0.3], [3., 2.0, 0.4], [3.0, 1.2, 0.5],
                        [2.6, 1.4, 0.6], [3.8, 1.8, 0.7], [3.6, 0.8, 0.8], [3.8, 2.2, 0.9], [3.4, 3.2, 1]]
    for degrade in dagradation_list:
        generate_mod_LR_bic(degrade[0], degrade[1], degrade[2])
