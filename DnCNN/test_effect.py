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
        self.num_epochs = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

config = DnCNNConfig()



patchs = gen_patches(config, config.dataset_path)
train_dataset = DatasetDnCNN(config, patchs)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

model = DnCNN(config).to(config.device)
model.load_state_dict(torch.load('./models/dncnn_sigma25.pth', map_location=config.device))

optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.MSELoss()

i = 3
x = train_dataset[i]['H'].unsqueeze(0).to(config.device)  # [1, C, H, W]
y = train_dataset[i]['L'].unsqueeze(0).to(config.device)

print(x.shape)
print(y.shape)

model.eval()
with torch.no_grad():
    out = model(y)
    out = out.squeeze().cpu().numpy()  # [H, W, C]
    y = y.squeeze().cpu().numpy()
    x = x.squeeze().cpu().numpy()

    x_pred = y - out
    show_img(y)
    # show_img(out)
    show_img(x_pred)
    show_img(x)
print(out.shape)
print(x.max(), x.min())
print(y.max(), y.min())
print(x_pred.max(), x_pred.min())
print(out.max(), out.min())

loss = criterion(torch.tensor(x_pred).to(config.device), torch.tensor(x).to(config.device))
psnr = compare_psnr(x, x_pred, data_range=1.0)
ssim = compare_ssim(x, x_pred, data_range=1.0)
print(f'Loss: {loss.item():.8f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}')