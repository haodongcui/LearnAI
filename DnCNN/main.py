import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from models import DnCNN
from dataset import DatasetDnCNN, gen_patches
from utils import show_img

# random seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DnCNNConfig:
    def __init__(self):
        self.dataset_path = './data/Set12/'  # path to high quality images
        self.n_channels = 1
        self.patch_size = 64
        self.stride = 30
        self.sigma = 25

        self.batch_size = 128
        self.num_workers = 0
        self.lr = 1e-3
        self.num_epochs = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)


config = DnCNNConfig()

patchs = gen_patches(config, config.dataset_path)
train_dataset = DatasetDnCNN(config, patchs)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

model = DnCNN(config).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.MSELoss()


for epoch in range(config.num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        img_L = batch['L'].to(config.device)  # [B, H, W, C] -> [B, C, H, W]
        img_H = batch['H'].to(config.device)
        noise = img_L - img_H

        optimizer.zero_grad()
        output = model(img_L)
        loss = criterion(output, noise)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    loss = epoch_loss / train_loader.dataset.__len__()
    psnr = compare_psnr(img_H.squeeze().cpu().numpy(), output.squeeze().detach().cpu().numpy(), data_range=1.0)
    print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss:.8f}, PSNR: {psnr:.2f}dB')

# Save the model checkpoint
torch.save(model.state_dict(), f'./models/dncnn_sigma{config.sigma}.pth')
print(f'Model saved to dncnn_sigma{config.sigma}.pth')