# DnCNN-Pytorch
DnCNN-Pytorch 复现
## 环境配置
```
conda create --name DnCNN-py38-torch241-cu124 python=3.8 -y
conda activate DnCNN-py38-torch241-cu124

pip install "F:\AI\DL\torch-whl\torch-2.4.1+cu124-cp38-cp38-win_amd64.whl"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install tensorboard
pip install tensorboardX
pip install opencv-python
pip install scikit-image
pip install matplotlib
pip install h5py
```

## 参考资料
- [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://ieeexplore.ieee.org/document/7839189)
- https://github.com/cszn/DnCNN
- https://github.com/SaoYan/DnCNN-PyTorch
- https://jishuzhan.net/article/1891261471598317569

## 注意
### patch
图像尺寸不同，可以resize统一尺寸，但其中涉及到插值会导致失真，而denoise对像素要求高，所以更精准的方法是将图像切割成小patch。
### rescale
图像像素值归一化到[0,1]，增加模型训练的稳定性。显示时再rescale到[0,255]。