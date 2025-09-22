# FFDNet
复现 FFDNet 快速灵活图像去噪

## 环境搭建
```
conda create --name FFDNet-py38-torch241-cu124 python=3.8
conda activate FFDNet-py38-torch241-cu124

pip install "F:\AI\DL\torch-whl\torch-2.4.1+cu124-cp38-cp38-win_amd64.whl"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install tensorboard
pip install tensorboardX
pip install scikit-image
pip install opencv-python
pip install matplotlib
pip install h5py
```

## 关键点
### 无损下采样
将空间信息转换为通道信息。具体步骤：
- 将图像划分为不重叠的2×2小块
- 将每个块中的4个像素值分配到通道维度
- 图像尺寸减半，但通道数变为4倍

### 图片尺寸不一致
分割不同patch? 还是直接填充边缘?

采用的分割patch