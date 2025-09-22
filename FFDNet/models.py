import torch
import torch.nn as nn


class FFDNet(nn.Module):
    def __init__(self, config):
        super(FFDNet, self).__init__()
        self.noise_level = 1
        self.dncnn = DnCNN(config)

    def forward(self, x):
        x = downsample(x)
        x = cat_with_noise(x, self.noise_level)
        x = self.dncnn(x)
        x = upsample(x)
        return x


class DnCNN(nn.Module):
    def __init__(self, config):
        super(DnCNN, self).__init__()
        self.in_channels = 3 * 4 + 1  # after downsample + noise
        self.out_channels = 3 * 4  # need to upsample
        self.hidden_channels = 64
        self.kernel_size = 3
        self.padding = 1
        self.num_of_layers = 15

        layers = []
        layers.append(nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.hidden_channels,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.hidden_channels,
                                    out_channels=self.hidden_channels,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.hidden_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.hidden_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dncnn(x)
        return x


def downsample(input):
    scale = 2
    scale2 = scale * scale  # 4
    N, C, H, W = input.size()

    C_out = C * scale2  # 4C
    H_out = H // scale  # H/2
    W_out = W // scale  # W/2
    idx_list = [[i, j] for j in range(scale) for i in range(scale)]  # [[0, 0], [0, 1], [1, 0], [1, 1]]

    out = torch.zeros(N, C_out, H_out, W_out, dtype=input.dtype, device=input.device)  # [N, 4C, H/2, W/2]
    for i, idx in enumerate(idx_list):
        out[:, i::scale2, :, :] = input[:, :, idx[0]::scale, idx[1]::scale]  # [N, C, H/2, W/2]
    return out

def upsample(input):
    scale = 2
    scale2 = scale * scale  # 4
    N, C, H, W = input.size()

    C_out = C // scale2  # C/4
    H_out = H * scale  # 2H
    W_out = W * scale  # 2W
    idx_list = [[i, j] for j in range(scale) for i in range(scale)]  # [[0, 0], [0, 1], [1, 0], [1, 1]]

    out = torch.zeros(N, C_out, H_out, W_out, dtype=input.dtype, device=input.device)  # [N, C/4, 2H, 2W]
    for i, idx in enumerate(idx_list):
        out[:, :, idx[0]::scale, idx[1]::scale] = input[:, i::scale2, :, :]  # [N, C/4, 2H, 2W]
    return out

def cat_with_noise(input, noise_level):
    N, C, H, W = input.size()
    noise = torch.FloatTensor(N, 1, H, W).fill_(noise_level).to(input.device)
    return torch.cat([input, noise], dim=1)  # [N, C+1, H, W]




if __name__ == "__main__":

    # 创建测试输入 (1×1×4×4)
    input = torch.arange(1, 17).view(1, 1, 4, 4).float()
    print("输入:\n", input)

    # 下采样测试
    down = downsample(input)
    print("\n下采样结果:\n", down)

    # 上采样测试
    up = upsample(down)
    print("\n上采样结果:\n", up)