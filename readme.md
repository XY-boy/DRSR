# D2U (INFFUS 2023)
### 📖[**Paper**](https://www.sciencedirect.com/science/article/pii/S1566253523001100) | 🖼️[**PDF**](/img/XY-IF.pdf)

PyTorch codes for "[From Degrade to Upgrade: Learning a Self-Supervised Degradation-Guided Adaptive Network for Blind Remote Sensing Image Super-Resolution](https://doi.org/10.1016/j.inffus.2023.03.021)", **Information Fusion**, 2023.

Authors: [Yi Xiao](https://xy-boy.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Kui Jiang](https://github.com/kuijiang94/home/blob/master/home.md), [Jiang He](https://jianghe96.github.io/), [Yuan Wang](https://scholar.google.com.hk/citations?user=lB1KOAcAAAAJ&hl), and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<br>
Wuhan University and Huawei Technology

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

## Dataset Preparation
**Step I.** Please download the following remote sensing datasets
| Data Type | [AID](https://captain-whu.github.io/AID/) | [DOTA-v2.0](https://captain-whu.github.io/DOTA/dataset.html) | [Jilin-1](https://ieeexplore.ieee.org/abstract/document/9530280) |
| :----: | :-----: | :----: | :----: |
|Training | [Download](https://onedrive.live.com/?authkey=%21AAqO0B6SeejPkr0&id=42EC9A19F3DE58D8%2176404&cid=42EC9A19F3DE58D8&parId=root&parQt=sharedby&o=OneUp) | None | None |
|Testing | [Download]() | [Download]() | [Download]() |

**Step II.** Prepare the test sets under different degradation settings

- For ***"Isotropic Blur"*** degradations:
Using the degradation function [`generate_mod_LR_bic.py`](https://github.com/yuanjunchai/IKC/blob/master/codes/scripts/generate_mod_LR_bic.py) in [IKC](https://github.com/yuanjunchai/IKC) by changing the kernel width σ∈[0.2,4.0] at line [`sig=2.0`](https://github.com/yuanjunchai/IKC/blob/2a846cf1194cd9bace08973d55ecd8fd3179fe48/codes/scripts/generate_mod_LR_bic.py#L95)
- For ***"Anisotropic Blur + Noise"*** degradations:
Using our modified funtion.
### Visual results on Isotropic Gaussian blur
 ![image](/img/res.png)
## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: xiao_yi@whu.edu.cn  
Tel: (+86) 15927574475 (WeChat)

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
Our work mainly borrows from [DASR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/DASR) and [SimCLR](https://github.com/sthalles/SimCLR). Thanks for these excellent works!
