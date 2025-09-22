import os.path
import numpy as np
import torch
import torch.utils.data as data
import cv2



class Dataset(data.Dataset):
    def __init__(self, img_pairs):
        super(Dataset, self).__init__()
        self.img_pairs = img_pairs

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index):
        rain_img = self.img_pairs[index]['rain']
        norain_img = self.img_pairs[index]['norain']
        return rain_img, norain_img


def prepare_data(dataset_dir, config):
    if config.load_prepared_data == True:
        return load_prepared_data(config.mid_data_dir)

    # 去掉不成对的图片
    rain_dict = {}
    norain_dict = {}
    for f in os.listdir(dataset_dir):
        if f.endswith('.png'):
            if f.startswith('rain-'):
                num = f[5:-4]  # 提取编号
                rain_dict[num] = os.path.join(dataset_dir, f)
            elif f.startswith('norain-'):
                num = f[7:-4]  # 提取编号
                norain_dict[num] = os.path.join(dataset_dir, f)
    common_nums = set(rain_dict.keys()) & set(norain_dict.keys())  # 取交集
    common_nums = sorted(common_nums, key=lambda x: int(x))  # 按数字排序
    common_nums = common_nums[:config.num_pairs]  # 只取前 num_pairs 对图像

    # 获得图片对
    img_pairs = []
    for num in common_nums:
        rain_img = read_img(rain_dict[num])
        norain_img = read_img(norain_dict[num])
        if rain_img is None or norain_img is None:  # 图片有损
            continue
        if rain_img.shape != norain_img.shape:  # 尺寸不一致
            continue
        img_pairs.append({'num': num, 'rain': rain_img, 'norain': norain_img})

    # 提取 patches 统一尺寸
    patch_pairs = []
    for pair in img_pairs:
        num, rain_img, norain_img = pair['num'], pair['rain'], pair['norain']
        i = 0
        rain_patches = extract_patches(rain_img, config.patch_size, config.stride)
        norain_patches = extract_patches(norain_img, config.patch_size, config.stride)
        for rain_patch, norain_patch in zip(rain_patches, norain_patches):
            patch_pairs.append({'num': num, 'rain': rain_patch, 'norain': norain_patch, 'patch_id': i})
            i += 1

    # 保存处理后的 patches
    if config.save_prepared_data == True:
        os.makedirs(config.mid_data_dir, exist_ok=True)
        data_path = os.path.join(config.mid_data_dir, 'patch_pairs.pt')
        torch.save(patch_pairs, data_path)
        print(f"Saved {len(patch_pairs)} patch_pairs to '{data_path}'")

    print(f"Found {len(patch_pairs)} valid patch_pairs from {len(img_pairs)} valid img_pairs in '{dataset_dir}'")
    return patch_pairs


# load prepared data
def load_prepared_data(prepared_data_dir):
    # 直接加载预处理好的数据
    data_path = os.path.join(prepared_data_dir, 'patch_pairs.pt')
    if os.path.exists(data_path):
        patch_pairs = torch.load(data_path)
        print(f"Loaded {len(patch_pairs)} patch_pairs from '{data_path}'")
        return patch_pairs
    else:
        print(f"Error: Prepared data file '{data_path}' not found.")
        return []


# 处理图片
def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        print(f"Error: Unable to read image at {path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = img.astype(np.float32) / 255.0  # 归一化到0-1
    img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
    return img


# 提取 patches 统一尺寸
def extract_patches(img, patch_size=50, stride=25):
    _, h, w = img.shape
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[:, i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return patches


# 展示图片
def show_img(img):
    import matplotlib.pyplot as plt
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)  # CHW to HWC
    plt.imshow(img)
    plt.axis('off')
    plt.show()