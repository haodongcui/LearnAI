## 环境

```bash
conda create -n Mario python=3.8
```
官方要求 python=3.8

但如果使用的是 python=3.10, 需要降级 numpy 到 numpy==1.26.0

```bash
pip install gym==0.23
pip install nes-py==8.1.8
pip install gym-super-mario-bros==7.4.0
pip install stable_baselines3==2.0.0
pip install Optuna
```

### 对于python=3.8, 安装 pytorch
对于3.10, 使用从[官网的torch列表](https://download.pytorch.org/whl/torch/)下载好的 whl 本地安装 pytorch
```bash
pip install "F:\DL\torch-whl\torch-2.4.1+cu124-cp38-cp38-win_amd64.whl"
```
再安装剩余包和依赖(cu126没有python=3.8的版本, 得用cu124)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 对于python=3.10, 安装 pytorch
对于3.10, 使用下载好的 whl 本地安装 pytorch
```bash
pip install "F:\DL\torch-whl\torch-2.6.0+cu126-cp310-cp310-win_amd64.whl"
```
再安装剩余包和依赖
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 主要参考
- gym_super_mario_bros的详细介绍 https://www.codeleading.com/article/50074885634/
- 可用的环境 https://zhuanlan.zhihu.com/p/693635138
- 游戏图像处理 https://github.com/Hjananggch/gym_super_mario