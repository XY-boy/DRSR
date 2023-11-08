# D2U (INFFUS 2023)
### 📖[**Paper**](https://www.sciencedirect.com/science/article/pii/S1566253523001100) | 🖼️[**PDF**](./img/XY-IF.pdf)

PyTorch codes for "[From Degrade to Upgrade: Learning a Self-Supervised Degradation-Guided Adaptive Network for Blind Remote Sensing Image Super-Resolution](https://www.sciencedirect.com/science/article/pii/S1566253523001100)", **Information Fusion**, 2023.

Authors: [Yi Xiao](https://xy-boy.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Kui Jiang](https://github.com/kuijiang94/home/blob/master/home.md), [Jiang He](https://jianghe96.github.io/), [Yuan Wang](https://scholar.google.com.hk/citations?user=lB1KOAcAAAAJ&hl), and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<br>
Wuhan University and Huawei Technology

### Abstract
>Over the past few years, single image super-resolution (SR) has become a hotspot in the remote sensing area, and numerous methods have made remarkable progress in this fundamental task. However, they usually rely on the assumption that images suffer from a fixed known degradation process, e.g., bicubic downsampling. To save us from performance drop when real-world distribution deviates from the naive assumption, blind image super-resolution for multiple and unknown degradations has been explored. Nevertheless, the lack of a real-world dataset and the challenge of reasonable degradation estimation hinder us from moving forward. In this paper, a self-supervised degradation-guided adaptive network is proposed to mitigate the domain gap between simulation and reality. Firstly, the complicated degradations are characterized by robust representations in embedding space, which promote adaptability to the downstream SR network with degradation priors. Specifically, we incorporated contrastive learning to blind remote sensing image SR, which guides the reconstruction process by encouraging the positive representations (relevant information) while punishing the negatives. Besides, an effective dual-wise feature modulation network is proposed for feature adaptation. With the guide of degradation representations, we conduct modulation on feature and channel dimensions to transform the low-resolution features into the desired domain that is suitable for reconstructing high-resolution images. Extensive experiments on three mainstream datasets have demonstrated our superiority against state-of-the-art methods. Our source code can be found at https://github.com/XY-boy/DRSR
>
### Network
 ![image](/img/D2U.jpg)
## 🧩Install
```
git clone https://github.com/XY-boy/DRSR.git
```
## Requirements
> - Python 3.8
> - PyTorch >= 1.9
> - Ubuntu 18.04, cuda-11.1

## Dataset Preparation (Offline)
**Step I.** Please download the following remote sensing datasets:
| Data Type | [AID](https://captain-whu.github.io/AID/) | [DOTA-v2.0](https://captain-whu.github.io/DOTA/dataset.html) | [Jilin-1](https://ieeexplore.ieee.org/abstract/document/9530280) |
| :----: | :-----: | :----: | :----: |
|Training | [Download](https://onedrive.live.com/?authkey=%21AAqO0B6SeejPkr0&id=42EC9A19F3DE58D8%2176404&cid=42EC9A19F3DE58D8&parId=root&parQt=sharedby&o=OneUp) | None | None |
|Testing | [Download]() | [Download]() | [Download]() |

**Step II.** Prepare the test sets under different degradation settings:

- For ***"Isotropic Blur"*** degradations:
Use the degradation function [`generate_mod_LR_bic.py`](https://github.com/yuanjunchai/IKC/blob/master/codes/scripts/generate_mod_LR_bic.py) in [IKC](https://github.com/yuanjunchai/IKC) by changing the kernel width σ∈[0.2,4.0] at line [`sig=2.0`](https://github.com/yuanjunchai/IKC/blob/2a846cf1194cd9bace08973d55ecd8fd3179fe48/codes/scripts/generate_mod_LR_bic.py#L95). Or using our function [`generate_mod_LR_bic_iso.py`](https://github.com/XY-boy/DRSR/blob/main/script/generate_mod_LR_bic_iso.py) by setting the kernel width list at line [`sig_list`](https://github.com/XY-boy/DRSR/blob/15ca57d11998a2e7ae3887ec761b395f0444ca85/script/generate_mod_LR_bic_iso.py#L94C24-L94C24).

- For ***"Anisotropic Blur + Noise"*** degradations:
Use our modified function [`generate_mod_LR_bic_aniso.py`](https://github.com/XY-boy/DRSR/blob/main/script/generate_mod_LR_bic_aniso.py) by changing the noise level at line [`noise_stable`](https://github.com/XY-boy/DRSR/blob/15ca57d11998a2e7ae3887ec761b395f0444ca85/script/generate_mod_LR_bic_aniso.py#L76C79-L76C79), then change the shape of anisotropic Gaussian blur kernel by setting λ1, λ2, and θ at line [`dagradation_list`](https://github.com/XY-boy/DRSR/blob/15ca57d11998a2e7ae3887ec761b395f0444ca85/script/generate_mod_LR_bic_aniso.py#L89).

## Usage
### Train
Set the training option at [`option/train.py`](https://github.com/XY-boy/DRSR/blob/main/options/train.py). Then run the main file:
```
python main.py
```
**Note**: The setting of isotropic Gaussian blur and anisotropic Gaussian blur are useless during model training.
### Test
- Download the pre-trained models from [checkpoint](https://github.com/XY-boy/DRSR/tree/main/checkpoint). We provide 4 weights for the evaluation of remote sensing and natural images!
```
d2u-aniso.pth/d2u-iso.pth    ----------    trained on remote sensing images (AID)
DRSR_Blur.pth/DRSR_Noisy.pth    -------    trained on natural images (DIV2K)
```
- For ***"Isotropic Blur"*** degradations: Change the `--sig` and other testing options at [`option/test.py`](https://github.com/XY-boy/DRSR/blob/main/options/train.py). Then run the test file:
```
python eval_iso.py
```
- For ***"Anisotropic Blur + Noise"*** degradations: Change the `noise`, `lr_folder`, `model_name`, and `save_results_dir` at [`eval_aniso.py`](https://github.com/XY-boy/DRSR/blob/main/eval_aniso.py). Then run the test file:
```
python eval_aniso.py
```
## Results
### Visual results on Isotropic Gaussian blur
 ![image](/img/res.png)
### Quantitative results on anisotropic Gaussian blur
 ![image](/img/res-aniso.png)
More Results can be found in our paper [**PDF**](/img/XY-IF.pdf)!
## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: xiao_yi@whu.edu.cn; xy574475@gmail.com

## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support！😊

```
@article{xiao2023d2u,
  title={From degrade to upgrade: Learning a self-supervised degradation guided adaptive network for blind remote sensing image super-resolution},
  author={Xiao, Yi and Yuan, Qiangqiang and Jiang, Kui and He, Jiang and Wang, Yuan and Zhang, Liangpei},
  journal={Information Fusion},
  volume={96},
  pages={297--311},
  year={2023},
  publisher={Elsevier}
}
```
## Acknowledgement
Our work mainly borrows from [DASR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/DASR) and [SimCLR](https://github.com/sthalles/SimCLR). Thanks to these excellent works!
