import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from models import FFDNet
from dataset import Dataset, prepare_data, show_img

class FFDNetConfig:
    def __init__(self):
        self.train_data_dir = './data/RainTrainH'
        self.test_data_dir = './data/RainTest'
        self.mid_data_dir = './data_prepared'
        self.patch_size = 300
        self.stride = 100
        self.save_prepared_data = True
        self.load_prepared_data = True
        self.num_pairs = 20  # 只取前 num_pairs 对图像, -1 表示全部

        self.noise_level = 25
        self.num_workers = 0
        self.lr = 1e-3
        self.batch_size = 32
        self.num_epochs = 10

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device:', self.device)

config = FFDNetConfig()

train_img_pairs = prepare_data(config.train_data_dir, config)
train_dataset = Dataset(train_img_pairs)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

val_img_pairs = prepare_data(config.test_data_dir, config)
val_dataset = Dataset(val_img_pairs)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

model = FFDNet(config).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.MSELoss()

best_val_psnr = 0.0

for epoch in range(config.num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        rain_img, norain_img = batch
        rain_img, norain_img = rain_img.to(config.device), norain_img.to(config.device)

        optimizer.zero_grad()
        out_img = model(rain_img)
        loss = criterion(out_img, norain_img)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    # train_psnr = compare_psnr(norain_img.cpu().numpy(), out_img.detach().cpu().numpy(), data_range=1)
    # train_ssim = compare_ssim(norain_img.cpu().numpy().squeeze(), out_img.detach().cpu().numpy().squeeze(), data_range=1, win_size=3)
    # print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {train_loss:.4f}, PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.4f}")

    # Validation
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            rain_img, norain_img = batch
            rain_img, norain_img = rain_img.to(config.device), norain_img.to(config.device)

            out_img = model(rain_img)
            loss = criterion(out_img, norain_img)
            val_epoch_loss += loss.item()

        val_loss = val_epoch_loss / len(val_loader)
        val_psnr = compare_psnr(norain_img.cpu().numpy(), out_img.detach().cpu().numpy(), data_range=1)
        val_ssim = compare_ssim(norain_img.cpu().numpy().squeeze(), out_img.detach().cpu().numpy().squeeze(), data_range=1, win_size=3)
        print(f"Validation Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
