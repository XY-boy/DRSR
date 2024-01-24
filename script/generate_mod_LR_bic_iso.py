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


def generate_mod_LR_bic(sig_num, clip_name):
    # set parameters
    up_scale = 4
    mod_scale = 4
    # set data dir
    sourcedir = 'H:/blind_VSR/test_video_GT/zhuhai/GT/' + clip_name #'/mnt/yjchai/SR_data/DIV2K_test_HR' #'/mnt/yjchai/SR_data/Flickr2K/Flickr2K_HR'
    savedir = 'H:/blind_VSR/IKC_test_set/zhuhai/sig'+str(sig_num) + '/' + clip_name #'/mnt/yjchai/SR_data/DIV2K_test' #'/mnt/yjchai/SR_data/Flickr2K_train'
    # set random seed
    util.set_random_seed(0)

    # load PCA matrix of enough kernel
    print('load PCA matrix')
    pca_matrix = torch.load('H:/blind_VSR/test_video_GT/pca_matrix.pth', map_location=lambda storage, loc: storage)
    print('PCA matrix shape: {}'.format(pca_matrix.shape))

    # saveHRpath = os.path.join(savedir, 'HR', 'x' + str(mod_scale))
    # saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    # saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))
    saveLRblurpath = os.path.join(savedir)

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    if not os.path.isdir(saveLRblurpath):
        os.mkdir(saveLRblurpath)
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
        sig = sig_num  # 用于制作stable iso

        # 制作训练集random=true，sig∈[0.2,4.0]；制作测试集的时候random=false，也就是sig固定
        # rate_iso=1，只考虑各向同性，rate_iso=0，只考虑各向异性
        # noise_high代表最大噪声，用来构建随机噪声训练集，设置为25/255=0.098，noise_stable用来构建测试集，水平为
        # 0 5 10 的时候，noise_stable = 0 5/255 10/255
        prepro = util.SRMDPreprocessing(up_scale, pca_matrix, random=False, para_input=10, kernel=21, noise=False,
                                        cuda=True, sig=sig, sig_min=0.2, sig_max=4.0, rate_iso=1, scaling=3,
                                        rate_cln=0.2, noise_high=0)
                                        # random(sig_min, sig_max) | stable kernel(sig)

        LR_img, ker_map = prepro(img_HR.view(1, C, H, W))
        image_LR_blur = util.tensor2img(LR_img)
        cv2.imwrite(os.path.join(saveLRblurpath, filename), image_LR_blur)  # 训练集由于sig是随机的，所以不用sig命名了，直接原始名字

        # print(ker_map.size())
        kernel_map_tensor[i] = ker_map
        torch.cuda.empty_cache()
    # save dataset corresponding kernel maps
    # torch.save(kernel_map_tensor, './AID_train_kermap_Aniso_G.pth')
    print("Image Blurring & Down smaple Done: X"+str(up_scale))

if __name__ == "__main__":
    sig_list = [0.8, 1.2, 1.6, 2.0]
    jilin_clip = ['000', '001', '002', '004', '005', '007']
    cblist = ['0208/', '0209/', '0210/', '0211/', '0212/', '0213/', '0214/', '0215/', '0216/', '0217/']
    uclist = ['0218/', '0219/', '0220/', '0221/', '0222/', '0223/', '0224/', '0225/', '0226/', '0227/', '0228/', '0229/']
    skylist = ['0230/', '0231/', '0232/', '0233/', '0234/', '0235/']
    zhuhai = ['0236/', '0237/', '0238/']
    for sig_num in sig_list:
        for clip in zhuhai  :
            generate_mod_LR_bic(sig_num=sig_num, clip_name=clip)
